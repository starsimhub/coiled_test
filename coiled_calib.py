from dask.distributed import wait
import starsim as ss
import sciris as sc
import pandas as pd
import numpy as np
import coiled

debug = False # If true, will run in serial
total_trials = [100, 10][debug]
n_agents = 2_000
do_plot = True

class CoiledCalibration(ss.Calibration):

    def calibrate(self, calib_pars=None, load=False, tidyup=True, **kwargs):
        """
        Perform calibration using dask/coiled

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            load (bool): whether to load existing trials from the database (if rerunning the same calibration)
            tidyup (bool): whether to delete temporary files from trial runs
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        import optuna as op

        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        self.run_args.update(kwargs) # Update optuna settings

        cluster = coiled.Cluster(
            n_workers=10,
            name='StarsimCalibrationOnCoiled',
        )
        client = cluster.get_client()

        # Run the optimization
        t0 = sc.tic()
        self.run_args.storage = op.integration.DaskStorage(storage=None, client=client)
        self.study = self.make_study()

        #self.run_workers() # <-- This is the line from the base class to run locally using multiprocessing
        futures = [
            client.submit(
                self.study.optimize,
                self.run_trial,
                n_trials=self.run_args.n_trials,
                pure=False
            )
            for _ in range(self.run_args.n_workers)
        ]

        wait(futures)
        #################################

        study = op.load_study(storage=self.run_args.storage, study_name=self.run_args.study_name, sampler=self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        self.sim_results = []
        if load:
            if self.verbose: print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results)
                    if tidyup:
                        try:
                            os.remove(filename)
                            if self.verbose: print(f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            if self.verbose: print(errormsg)
                    if self.verbose: print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    if self.verbose: print(errormsg)

        # Compare the results
        self.parse_study(study)

        if self.verbose: print('Best pars:', self.best_pars)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()

        return self

    # dummy run_trial function --> also triggers the error
    # def run_trial(self, *args, **kwargs):
    #     return 0


def make_sim():
    sir = ss.SIR(
        beta = ss.beta(0.075),
        init_prev = ss.bernoulli(0.02),
    )
    random = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(
        n_agents = n_agents,
        start = sc.date('2020-01-01'),
        stop = sc.date('2020-02-12'),
        dt = 1,
        unit = 'day',
        diseases = sir,
        networks = random,
        verbose = 0,
    )

    return sim


def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_reps', 1)

    sir = sim.pars.diseases # There is only one disease in this simulation and it is a SIR
    net = sim.pars.networks # There is only one network in this simulation and it is a RandomNet

    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if k == 'beta':
            sir.pars.beta = ss.beta(v)
        elif k == 'init_prev':
            sir.pars.init_prev = ss.bernoulli(v)
        elif k == 'n_contacts':
            net.pars.n_contacts = ss.poisson(v)
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    if reps == 1:
        return sim

    # Ignoring the random seed if provided via the reseed=True option in Calibration
    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, reps)), initialize=True, debug=True, parallel=False)
    return ms


def extract_prevalence(sim):
    df = pd.DataFrame({
            'x': sim.results.sir.n_infected, # Instead of prevalence, let's compute it from infected and n_alive
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t'))
    return df

def test_coiled(do_plot=True):
    sc.heading('Testing a single parameter (beta) with a normally distributed likelihood, one data point')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True),
    )

    # Make the sim and data
    sim = make_sim()

    prevalence = ss.Normal(
        name = 'Disease prevalence',
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': [0.13],    # Prevalence of infection
        }, index=pd.Index([ss.date('2020-01-12')], name='t')), # On these dates

        #expected = pd.DataFrame({
        #    'x': [0.13, 0.16, 0.06],    # Prevalence of infection
        #}, index=pd.Index([ss.date(d) for d in ['2020-01-12', '2020-01-25', '2020-02-02']], name='t')), # On these dates

        #expected = pd.DataFrame({
        #    'x':  [740], # Number of new infections
        #    'n':  [100],   # Number of person-years
        #    't':  [ss.date(d) for d in ['2020-01-07']], # Between t and t1
        #    't1': [ss.date(d) for d in ['2020-01-08']],
        #}).set_index(['t', 't1']),
        
        extract_fn = extract_prevalence,
        #extract_fn = lambda sim: pd.DataFrame({
        #    'x': sim.results.sir.n_infected, # Instead of prevalence, let's compute it from infected and n_alive
        #    'n': sim.results.n_alive,
        #}, index=pd.Index(sim.results.timevec, name='t')),

        sigma2 = 0.05, # (num_replicates/sigma2_model + 1/sigma2_data)^-1
    )

    # Make the calibration
    calib = CoiledCalibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim,
        build_kw = dict(n_reps=5), # Reps per point
        reseed = False,
        components = [prevalence],
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(do_plot=False), 'Calibration did not improve the fit'

    # Call plotting to look for exceptions
    if do_plot:
        calib.plot_final()
        calib.plot(bootstrap=False)
        calib.plot(bootstrap=True)
        calib.plot_optuna(['plot_param_importances', 'plot_optimization_history'])

    return sim, calib

if __name__ == '__main__':
    test_coiled()