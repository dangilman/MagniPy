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

    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=None, identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                            source_shape='GAUSSIAN', source_size=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={},raytrace_with=None):


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

        lens_systems = []

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                for real in realizations:
                    lens_systems.append(self.build_system(main=macromodel, additional_halos=real, multiplane=multiplane,
                                                          filter_by_pos=filter_by_pos,**filter_kwargs))
            else:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=None,multiplane=multiplane))

        else:

            lens_systems.append(copy.deepcopy(full_system))

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        data = self.solve_4imgs(lens_systems=lens_systems, method=method, identifier=identifier, srcx=srcx, srcy=srcy,
                                grid_rmax=grid_rmax,
                                res=res, source_shape=source_shape, ray_trace=ray_trace, raytrace_with=raytrace_with, source_size=source_size)

        if sort_by_pos is not None:
            data[0].sort_by_pos(sort_by_pos.x,sort_by_pos.y)
        return data


    def two_step_optimize(self, macromodel=None, datatofit=None, realizations=None, multiplane=False, method=None, ray_trace=None, sigmas=None,
                          identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                          source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None,
                          filter_by_position=False, polar_grid=False, filter_kwargs={},solver_type='PROFILE_SHEAR'):

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
                                                           filter_by_position=filter_by_position, filter_kwargs=filter_kwargs)
        else:
            pass
            #_, macromodel_init = self.macromodel_initialize(macromodel=copy.deepcopy(macromodel),datatofit=datatofit,
            #                                                multiplane=multiplane, method='lenstronomy', ray_trace=ray_trace,
            #                                                sigmas=sigmas,srcx=srcx, srcy=srcy,
            #                                                grid_rmax=grid_rmax, res=res,
            #                                                source_shape=source_shape, source_size=source_size,
            #                                                print_mag=print_mag,
            #                                                filter_by_position=filter_by_position,
            #                                                filter_kwargs=filter_kwargs,solver_type=solver_type)

        optimized_data, newsystem = self.fit(macromodel=macromodel, datatofit=datatofit, realizations=realizations, multiplane=multiplane, method=method,
                                                       ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx, srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                       source_shape=source_shape, source_size=source_size, print_mag=print_mag, raytrace_with=raytrace_with,
                                                       filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                    filter_kwargs=filter_kwargs,solver_type=solver_type)


        return optimized_data,newsystem

    def macromodel_initialize(self, macromodel, datatofit, multiplane, method=None, ray_trace=None, sigmas=None,
                              identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                              source_shape='GAUSSIAN', source_size=None, print_mag=False, filter_by_position=False,
                              filter_kwargs={},solver_type=None):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        if sigmas is None:
            sigmas = default_sigmas

        if method is None:
            method = default_solve_method

        lens_systems = []

        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), additional_halos=None, multiplane=multiplane,
                                              filter_by_position=filter_by_position,**filter_kwargs))

        optimized_data, model = self.optimize_4imgs(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                    sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                    res=res, source_shape=source_shape, ray_trace=False,
                                                    source_size=source_size, print_mag=print_mag, opt_routine='randomize',solver_type=solver_type)

        newmacromodel = model[0].lens_components[0]

        return optimized_data,newmacromodel

    def fit(self, macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, filter_by_position=False, polar_grid=False,
                      filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR'):

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


        optimized_data, model = self.optimize_4imgs(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                    sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                    res=res, source_shape=source_shape, ray_trace=ray_trace,
                                                    source_size=source_size, print_mag=print_mag, opt_routine=basic_or_full,
                                                    raytrace_with=raytrace_with, polar_grid=polar_grid,solver_type=solver_type)

        return optimized_data,model

    def do_raytrace(self,lens_system,data=None,multiplane=None,grid_rmax=None,res=None,source_shape=None,source_size=None,
                    raytrace_with=None,srcx=None,srcy=None,build=False,realizations=None):

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

    def raytrace_images(self, full_system=None, macromodel=None, xcoord=None, ycoord = None, realizations=None, multiplane=None,
                        identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                        source_shape='GAUSSIAN', source_size=None, filter_by_position=False,
                        image_index=None):

        lens_systems = []

        assert image_index is not None

        if full_system is None:

            assert macromodel is not None
            if realizations is not None:
                assert len(realizations) == 1
                for real in realizations:
                    lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), additional_halos=real, multiplane=multiplane))

        else:
            lens_systems.append(copy.deepcopy(full_system))


        trace = RayTrace(xsrc=srcx, ysrc=srcy, multiplane=multiplane, method='lensmodel', grid_rmax=grid_rmax, res=res,
                         source_shape=source_shape,
                         cosmology=self.cosmo, source_size=source_size)

        magnifications, image = trace.get_images(xpos=[xcoord], ypos=[ycoord], lens_system=lens_systems[0],
                                                 return_image=True,which_image=image_index)

        return magnifications, image










