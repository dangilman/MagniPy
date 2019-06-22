from MagniPy.Workflow.macro_mcmc import MacroMCMC

def execute_mcmc(fname, lens_class, N, optkwargs, to_vary = {}, return_physical_positions=True):

    mcmc = MacroMCMC(lens_class.data, lens_class.optimize_fit)
    mcmc.from_class = True

    tovary = {'source_size_kpc': [lens_class.srcmin, lens_class.srcmax],
              'gamma': [lens_class.gamma_min, lens_class.gamma_max]}

    for key in to_vary.keys():
        tovary.update({key: to_vary[key]})

    if hasattr(lens_class, 'satellite_mass_model'):

        satellite_mass_model = lens_class.satellite_mass_model
        satellite_kwargs = lens_class.satellite_kwargs
        satellite_redshift = lens_class.satellite_redshift
        satellite_convention = lens_class.satellite_convention

    else:

        satellite_mass_model = None
        satellite_kwargs = None
        satellite_redshift = None
        satellite_convention = None

    _, _ = mcmc.run(flux_ratio_index=lens_class.flux_ratio_index, macromodel=lens_class._macromodel,
                      N=N, optimizer_kwargs=optkwargs,
                            tovary=tovary, write_to_file=True, fname=fname,
                              satellite_mass_model=satellite_mass_model,satellite_kwargs = satellite_kwargs,
                    satellite_convention=satellite_convention,satellite_redshift=satellite_redshift,
                    return_physical_positions=return_physical_positions,z_source=lens_class.zsrc)

from MagniPy.Workflow.grism_lenses.rxj0911 import RXJ0911
from MagniPy.Workflow.grism_lenses.lens1606 import Lens1606
from MagniPy.Workflow.grism_lenses.b1422 import Lens1422
from MagniPy.Workflow.grism_lenses.rxj1131 import Lens1131

simple_quads = [Lens1131()]
N=400
optimizer_kwargs = [{'tol_mag': None, 'pso_compute_magnification': 1e+2,
                    'n_particles': 30, 'verbose': False,
                    'optimize_routine': 'fixedshearpowerlaw', 'grid_res': 0.001, 'multiplane': True},
                    ]

#xsat = simple_quads[0].satellite_kwargs[0]['center_x']
#ysat = simple_quads[0].satellite_kwargs[0]['center_y']

varyparams = [{'fixed_param_uniform': {'shear':[0.09,0.17]}}]

fnames = ['./quad_mcmc/1131_samples_fixshear.txt']

for fname, lens, opt_kwargs_i, to_vary_i in \
        zip(fnames, simple_quads, optimizer_kwargs, varyparams):

    execute_mcmc(fname, lens, N, opt_kwargs_i, to_vary = to_vary_i)

