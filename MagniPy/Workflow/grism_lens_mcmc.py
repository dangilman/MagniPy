from MagniPy.Workflow.macro_mcmc import MacroMCMC

def execute_mcmc(fname, lens_class, N, optkwargs, to_vary = {}):

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

    _, _ = mcmc.run(flux_ratio_index=lens_class.flux_ratio_index, macromodel=lens_class._macromodel,
                      N=N, optimizer_kwargs=optkwargs,
                            tovary=tovary, write_to_file=True, fname=fname,
                              satellite_mass_model=satellite_mass_model, satellite_kwargs = satellite_kwargs)

from MagniPy.Workflow.grism_lenses.J0405 import J0405
from MagniPy.Workflow.grism_lenses.lens1606 import Lens1606
from MagniPy.Workflow.grism_lenses.lens2026 import Lens2026
from MagniPy.Workflow.grism_lenses.lens2033 import WFI2033
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038
from MagniPy.Workflow.grism_lenses.rxj0911 import RXJ0911
from MagniPy.Workflow.grism_lenses.lens1330 import Lens1330

from MagniPy.Workflow.grism_lenses.lens0810 import Lens0810

#simple_quads = [Lens2038(), Lens2026(), J0405(), Lens1606()]
simple_quads = [Lens1330(), Lens1606(), Lens1330()]
N = 1000

optimizer_kwargs = [{'tol_mag': 0.2, 'pso_compute_magnification': 1e+5,
                    'n_particles': 50, 'verbose': False,
                    'optimize_routine': 'fixed_powerlaw_shear',
                     'use_finite_source': False},
                    {'tol_mag': 0.2, 'pso_compute_magnification': 1e+5,
                     'n_particles': 50, 'verbose': False,
                     'optimize_routine': 'fixed_powerlaw_shear',
                     },{'tol_mag': 0.2, 'pso_compute_magnification': 1e+5,
                     'n_particles': 50, 'verbose': False,
                     'optimize_routine': 'fixed_powerlaw_shear',
                     }]

lensJ1606_sat = simple_quads[1].satellite_kwargs[0]
sat_thetaE1606 = lensJ1606_sat['theta_E']
sat_centerx1606 = lensJ1606_sat['center_x']
sat_centery1606 = lensJ1606_sat['center_y']

# 2033: wider prior on power law slope (to smaller values)

#For satellites
# impose Gaussian prior on position
# wider flat prior on Einstein radius

varyparams = [{'satellite_keff': [0, 0.35], 'satellite_reff': [0.1, 0.7],
               'satellite_ellip': [0.6, 0.9], 'satellite_PA': [-75, -45]},
              {'satellite_x': [sat_centerx1606, 0.1],
               'satellite_y': [sat_centery1606, 0.1],
               'satellite_theta_E': [0, sat_thetaE1606 + 0.6]}, {}]

fnames = ['./quad_mcmc/1330_samples.txt',
          './quad_mcmc/1606_samples.txt',
          './quad_mcmc/1330nodisk_samples.txt']

for fname, lens, opt_kwargs_i, to_vary_i in zip(fnames, simple_quads, optimizer_kwargs, varyparams):

    if fname == fnames[0]:
        continue
    execute_mcmc(fname, lens, N, opt_kwargs_i, to_vary = to_vary_i)

