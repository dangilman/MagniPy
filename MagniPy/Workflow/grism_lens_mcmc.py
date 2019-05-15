from MagniPy.Workflow.macro_mcmc import MacroMCMC

def execute_mcmc_1606(fname, lens_class, N):

    mcmc = MacroMCMC(lens_class.data, lens_class.optimize_fit)
    mcmc.from_class = True

    tovary = {'source_size_kpc': [lens_class.srcmin, lens_class.srcmax],
              'gamma': [lens_class.gamma_min, lens_class.gamma_max]}

    if hasattr(lens_class, 'satellite_mass_model'):

        satellite_mass_model = lens_class.satellite_mass_model
        satellite_kwargs = lens_class.satellite_kwargs[0]

    else:

        satellite_mass_model = None
        satellite_kwargs = None

    _, _ = mcmc.run(flux_ratio_index=0, macromodel=lens_class._macromodel,
                      N=N, optimizer_kwargs=optimizer_kwargs,
                            tovary=tovary, write_to_file=True, fname=fname,
                              satellite_mass_model=satellite_mass_model, satellite_kwargs = satellite_kwargs)

from MagniPy.Workflow.grism_lenses.lens1606 import Lens1606

N = 100
optimizer_kwargs = {'tol_mag': None, 'pso_compute_magnification': 1e+9,
                    'n_particles': 50, 'verbose': False}
execute_mcmc_1606('1606_nofluxes.txt', Lens1606(), N)

optimizer_kwargs = {'tol_mag': 0.05, 'pso_compute_magnification': 1e+9,
                    'n_particles': 50, 'verbose': False}
execute_mcmc_1606('1606_withfluxes.txt', Lens1606(), N)
