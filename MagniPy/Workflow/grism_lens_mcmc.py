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
                    satellite_convention=satellite_convention,satellite_redshift=satellite_redshift)


from MagniPy.Workflow.grism_lenses.he0435 import Lens0435

simple_quads = [Lens0435()]
N = 500

optimizer_kwargs = [{'tol_mag': 0.3, 'pso_compute_magnification': 1e+2,
                    'n_particles': 30, 'verbose': False,
                    'optimize_routine': 'fixed_powerlaw_shear', 'grid_res': 0.001},
                    ]

# 2033: wider prior on power law slope (to smaller values)

#For satellites
# impose Gaussian prior on position
# wider flat prior on Einstein radius

xsat = simple_quads[0].satellite_kwargs[0]['center_x']
ysat = simple_quads[0].satellite_kwargs[0]['center_y']

varyparams = [{'satellite_theta_E': [0.0, 0.7], 'satellite_x': [xsat, 0.05],
               'satellite_y': [ysat, 0.05]}]

fnames = ['./quad_mcmc/0435_samples.txt']

for fname, lens, opt_kwargs_i, to_vary_i in \
        zip(fnames, simple_quads, optimizer_kwargs, varyparams):

    execute_mcmc(fname, lens, N, opt_kwargs_i, to_vary = to_vary_i)

