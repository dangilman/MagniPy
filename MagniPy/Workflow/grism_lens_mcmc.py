from MagniPy.Workflow.macro_mcmc import MacroMCMC

def execute_mcmc(fname, lens_class, N, to_vary = {}):

    mcmc = MacroMCMC(lens_class.data, lens_class.optimize_fit)
    mcmc.from_class = True

    tovary = {'source_size_kpc': [lens_class.srcmin, lens_class.srcmax],
              'gamma': [lens_class.gamma_min, lens_class.gamma_max]}

    for key in to_vary.keys():
        tovary.update({key: to_vary[key]})

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
from MagniPy.Workflow.grism_lenses.lens2026 import Lens2026
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038

N = 200
optimizer_kwargs = {'tol_mag': 0.1, 'pso_compute_magnification': 1e+9,
                    'n_particles': 50, 'verbose': False, 'optimize_routine': 'fixedshearpowerlaw'}

execute_mcmc('2038_fixedshear.txt', Lens2038(), N, to_vary={'fixed_param_gaussian': {'shear': [0.09, 0.02]}})
#execute_mcmc('2026_fixedshear.txt', Lens2026(), N, to_vary={'fixed_param_uniform': {'shear': [0.05, 0.22]}})
#execute_mcmc('1606_fixedshear.txt', Lens1606(), N, to_vary={'fixed_param_gaussian': {'shear': [0.16, 0.025]}})

if False:
    lenses = [Lens2026(), Lens2038(), Lens1606()]
    names = [['2026_nofluxes.txt', '2026_withfluxes.txt'], ['2038_nofluxes.txt', '2038_withfluxes.txt']]

    for j, lens in enumerate(lenses):

        N = 200
        optimizer_kwargs = {'tol_mag': None, 'pso_compute_magnification': 1e+9,
                            'n_particles': 50, 'verbose': False}
        execute_mcmc(names[j][0], lens, N)

        optimizer_kwargs = {'tol_mag': 0.05, 'pso_compute_magnification': 1e+9,
                            'n_particles': 50, 'verbose': False}
        execute_mcmc(names[j][1], lens, N)

