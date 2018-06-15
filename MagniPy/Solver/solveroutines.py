from MagniPy.magnipy import Magnipy
from RayTrace.raytrace import RayTrace
from MagniPy.LensBuild.defaults import raytrace_with_default,default_sigmas,default_gridrmax,default_res,\
    default_source_shape,default_source_size,default_solve_method,default_file_identifier
import copy
from MagniPy.Solver.LenstronomyWrap.generate_input import LenstronomyWrap

class SolveRoutines(Magnipy):
    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def optimize_4imgs_lenstronomy(self,macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      grid_rmax=None, res=None,source_shape='GAUSSIAN', source_size=None,raytrace_with=None,
                      polar_grid=True,run_mode = 'src_plane_chi2',solver_type='PROFILE_SHEAR',n_particles=300,
                      n_iterations=100,tol_source=0.000001,tol_centroid=0.01,tol_mag=None,centroid_0=[0,0]):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        lens_systems= []

        ################################################################################

        # If macromodel is a list same length as realizations, build the systems and fit each one

        if isinstance(macromodel,list):

            assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

            for macro,real in zip(macromodel,realizations):
                lens_systems.append(self.build_system(main=macro,additional_halos=real,multiplane=multiplane))
        else:
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=real,
                                                          multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))

        _fit_data, fit_systems = self._solve_4imgs_lenstronomy(lens_systems=lens_systems, data2fit=datatofit, grid_rmax=grid_rmax,
                                                                  res=res, source_shape=source_shape, source_size=source_size, polar_grid=polar_grid,
                                                                  raytrace_with=raytrace_with, solver_type=solver_type,
                                                                  print_mag=False, N_iter_max=100)
        print 'with fit only:',np.sum(0.5*(_fit_data[0].m-datatofit.m)**2)*0.2**-1

        optimized_data, model = self._optimize_4imgs_lenstronomy(lens_systems=fit_systems, data2fit=datatofit, tol_source=tol_source,
                                  tol_mag=tol_mag, tol_centroid=tol_centroid,centroid_0=centroid_0, n_particles=n_particles,
                                  n_iterations=n_iterations, run_mode=run_mode,grid_rmax=grid_rmax, res=res,
                                 source_size=source_size,raytrace_with=raytrace_with,
                                 source_shape=source_shape,polar_grid=polar_grid, solver_type=solver_type)

        return optimized_data,model



    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=True, identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                            source_shape='GAUSSIAN', source_size=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={},raytrace_with=None,polar_grid=True,shr_coords=1):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems = []

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=macromodel, additional_halos=real, multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=None,multiplane=multiplane))

        else:

            lens_systems.append(copy.deepcopy(full_system))

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        data = self._solve_4imgs(lens_systems=lens_systems, method=method, identifier=identifier, srcx=srcx, srcy=srcy,
                                 grid_rmax=grid_rmax, res=res, source_shape=source_shape, ray_trace=ray_trace,
                                 raytrace_with=raytrace_with, source_size=source_size, polar_grid=polar_grid, shr_coords=shr_coords)

        if sort_by_pos is not None:
            data[0].sort_by_pos(sort_by_pos.x,sort_by_pos.y)
        return data


    def two_step_optimize(self, macromodel=None, datatofit=None, realizations=None, multiplane=False, method=None, ray_trace=None, sigmas=None,
                          identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                          source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None,
                          filter_by_position=False, polar_grid=True, filter_kwargs={},solver_type='PROFILE_SHEAR',
                          N_iter_max=100,shr_coords=1):

        # optimizes the macromodel first, then uses it to optimize with additional halos in the lens model

        assert datatofit is not None

        if sigmas is None:
            sigmas = default_sigmas

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        if method == 'lensmodel':
            _,macromodel_init = self.macromodel_initialize(macromodel=copy.deepcopy(macromodel), datatofit=datatofit,
                                                           multiplane=multiplane, method='lensmodel', ray_trace=ray_trace, sigmas=sigmas,
                                                           identifier=identifier, srcx=srcx, srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                           source_shape=source_shape, source_size=source_size, print_mag=print_mag,
                                                           filter_by_position=filter_by_position, filter_kwargs=filter_kwargs,shr_coords=shr_coords)
            optimized_data, newsystem = self.fit(macromodel=macromodel_init, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy, grid_rmax=grid_rmax, res=res,method='lensmodel',
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs,shr_coords=shr_coords)
        else:

            optimized_data, newsystem = self.fit(macromodel=macromodel, datatofit=datatofit,
                                                 realizations=realizations, multiplane=multiplane, method=method,
                                                 ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx,
                                                 srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                 source_shape=source_shape, source_size=source_size,
                                                 print_mag=print_mag, raytrace_with=raytrace_with,
                                                 filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                 filter_kwargs=filter_kwargs, solver_type=solver_type,N_iter_max=N_iter_max)




        return optimized_data,newsystem

    def macromodel_initialize(self, macromodel, datatofit, multiplane, method=None, ray_trace=None, sigmas=None,
                              identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                              source_shape='GAUSSIAN', source_size=None, print_mag=False, filter_by_position=False,
                              filter_kwargs={},solver_type=None,shr_coords=1):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        lens_systems = []

        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), additional_halos=None, multiplane=multiplane))

        optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                               sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                               res=res, source_shape=source_shape, ray_trace=False,
                                                               source_size=source_size, print_mag=print_mag, opt_routine='randomize',
                                                               solver_type=solver_type, shr_coords=shr_coords)

        newmacromodel = model[0].lens_components[0]

        return optimized_data,newmacromodel

    def fit(self, macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, filter_by_position=False,
                      polar_grid=True,filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR',N_iter_max=100,shr_coords=1):

        # uses source plane chi^2


        if which_chi == 'src':
            basic_or_full = 'basic'
        else:
            basic_or_full = 'full'
        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if method is None:
            method = default_solve_method

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        if identifier is None:
            identifier = default_file_identifier

        lens_systems= []

        ################################################################################

        # If macromodel is a list same length as realizations, build the systems and fit each one

        if isinstance(macromodel,list):

            assert len(macromodel) == len(realizations), 'if macromodel is a list, must have same number of elements as realizations'

            for macro,real in zip(macromodel,realizations):
                lens_systems.append(self.build_system(main=macro,additional_halos=real,multiplane=multiplane))
        else:
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=real,
                                                          multiplane=multiplane))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))

        if method == 'lenstronomy':
            optimized_data, model = self._solve_4imgs_lenstronomy(lens_systems=lens_systems, data2fit=datatofit, grid_rmax=grid_rmax,
                                                                  res=res, source_shape=source_shape, source_size=source_size, polar_grid=polar_grid,
                                                                  raytrace_with=raytrace_with, solver_type=solver_type,
                                                                  print_mag=print_mag, N_iter_max=N_iter_max)

        else:
            optimized_data, model = self._optimize_4imgs_lensmodel(lens_systems=lens_systems, data2fit=datatofit,
                                                                   method=method,
                                                                   sigmas=sigmas, identifier=identifier,
                                                                   grid_rmax=grid_rmax,
                                                                   res=res, source_shape=source_shape,
                                                                   ray_trace=ray_trace,
                                                                   source_size=source_size, print_mag=print_mag,
                                                                   opt_routine=basic_or_full,
                                                                   raytrace_with=raytrace_with, polar_grid=polar_grid,
                                                                   solver_type=solver_type, shr_coords=shr_coords)

        return optimized_data,model

    def _trace(self, lens_system, data=None, multiplane=None, grid_rmax=None, res=None, source_shape=None, source_size=None,
               raytrace_with=None, srcx=None, srcy=None, build=False, realizations=None):

        if raytrace_with is None:
            raytrace_with = raytrace_with_default

        if grid_rmax is None:
            grid_rmax = default_gridrmax(srcsize=source_size)

        if source_shape is None:
            source_shape = default_source_shape

        if source_size is None:
            source_size = default_source_size

        if res is None:
            res = default_res(source_size)

        if build:
            lens_system = self.build_system(main=lens_system,realizations=realizations,multiplane=multiplane)

        if raytrace_with == 'lensmodel':

            xsrc,ysrc = data.srcx,data.srcy

            if srcx is not None:
                xsrc = srcx
            if srcy is not None:
                ysrc = srcy

            fluxes = self.do_raytrace_lensmodel(lens_system=lens_system,xpos=data.x,ypos=data.y,xsrc=xsrc,ysrc=ysrc,
                                    multiplane=multiplane,grid_rmax=grid_rmax,res=res,source_shape=source_shape,source_size=source_size,
                                       cosmology=self.cosmo,zsrc=self.cosmo.zsrc,raytrace_with=raytrace_with)

        elif raytrace_with == 'lenstronomy':

            lenstronomy_wrap = LenstronomyWrap(multiplane=multiplane,cosmo=self.cosmo,z_source=self.cosmo.zsrc)
            lenstronomy_wrap.assemble(lens_system)

            fluxes = self.do_raytrace_lenstronomy(lenstronomy_wrap,data.x,data.y,multiplane=multiplane,gridsize=grid_rmax,res=res,
                                         source_shape=source_shape,source_size=source_size,cosmology=self.cosmo.cosmo,zsrc=self.cosmo.zsrc)

        return fluxes*max(fluxes)**-1

if False:
    from MagniPy.Analysis.PresetOperations.halo_constructor import Realization
    from MagniPy.LensBuild.defaults import *
    from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
    from MagniPy.util import cart_to_polar

    lens_params = {'R_ein':1.04,'x':0.001,'y':0.00,'ellip':0.3,'ellip_theta':10,'shear':0.045,'shear_theta':34,'gamma':2}
    lens_params_opt = {'R_ein':0.9,'x':0.00,'y':0.00,'ellip':0.1,'ellip_theta':60,'shear':0.045,'shear_theta':70,'gamma':2}
    start = Deflector(subclass=SIE(),redshift=0.5,tovary=True,varyflags=['1','1','1','1','1','1','1','0','0','0'],
                      **lens_params)
    start_opt = Deflector(subclass=SIE(),redshift=0.5,tovary=True,varyflags=['1','1','1','1','1','1','1','0','0','0'],
                      **lens_params_opt)

    real = Realization(0.5,1.5)

    halos = real.halo_constructor('TNFW','plaw_main',{'fsub':0.01},Nrealizations=1)
    solver = SolveRoutines(0.5,1.5)

    dtofit = solver.solve_lens_equation(macromodel=start,realizations=None,multiplane=False,srcx=0.03,srcy=-0.04)

    opt,mod = solver.optimize_4imgs_lenstronomy(datatofit=dtofit[0],realizations=halos,macromodel=start_opt,
                                                multiplane=False,tol_source=1e-8,tol_mag=0.01,n_particles=100,n_iterations=20)














