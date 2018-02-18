from MagniPy.magnipy import Magnipy
from RayTrace.raytrace import RayTrace
import copy

class SolveRoutines(Magnipy):
    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def solve_lens_equation(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                            ray_trace=None, identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                            source_shape='GAUSSIAN', source_size=None, sort_by_pos=None, filter_subhalos=False,
                            filter_by_pos=False, filter_kwargs={}):


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
                                res=res, source_shape=source_shape, ray_trace=ray_trace, source_size=source_size)

        if sort_by_pos is not None:
            data[0].sort_by_pos(sort_by_pos.x,sort_by_pos.y)
        return data


    def two_step_optimize(self, macromodel=None, datatofit=None, realizations=None, multiplane=False, method=None, ray_trace=None, sigmas=None,
                          identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                          source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None,
                          filter_by_position=False, polar_grid=False, filter_kwargs={},solver_type='PROFILE_SHEAR'):

        # optimizes the macromodel first, then uses it to optimize with additional halos in the lens model
        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        if method == 'lensmodel':
            _,macromodel_init = self.macromodel_initialize(macromodel=copy.deepcopy(macromodel), datatofit=datatofit,
                                                           multiplane=multiplane, method=method, ray_trace=ray_trace, sigmas=sigmas,
                                                           identifier=identifier, srcx=srcx, srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                           source_shape=source_shape, source_size=source_size, print_mag=print_mag,
                                                           filter_by_position=filter_by_position, filter_kwargs=filter_kwargs)
        else:
            macromodel_init = macromodel


        optimized_data, newsystem = self.fit(macromodel=macromodel_init, datatofit=datatofit, realizations=realizations, multiplane=multiplane, method=method,
                                                       ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx, srcy=srcy, grid_rmax=grid_rmax, res=res,
                                                       source_shape=source_shape, source_size=source_size, print_mag=print_mag, raytrace_with=raytrace_with,
                                                       filter_by_position=filter_by_position, polar_grid=polar_grid,
                                                    filter_kwargs=filter_kwargs,solver_type=solver_type)


        return optimized_data,newsystem

    def macromodel_initialize(self, macromodel, datatofit, multiplane, method=None, ray_trace=None, sigmas=None,
                              identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                              source_shape='GAUSSIAN', source_size=None, print_mag=False, filter_by_position=False,
                              filter_kwargs={}):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        lens_systems = []

        lens_systems.append(self.build_system(main=copy.deepcopy(macromodel), additional_halos=None, multiplane=multiplane,
                                              filter_by_position=filter_by_position,**filter_kwargs))

        optimized_data, model = self.optimize_4imgs(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                    sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                    res=res, source_shape=source_shape, ray_trace=False,
                                                    source_size=source_size, print_mag=print_mag, opt_routine='randomize')
        newmacromodel = model[0].lens_components[0]

        return optimized_data,newmacromodel

    def fit(self, macromodel=None, datatofit=None, realizations=None, multiplane = None, method=None, ray_trace=True, sigmas=None,
                      identifier=None, srcx=None, srcy=None, grid_rmax=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False, raytrace_with=None, filter_by_position=False, polar_grid=False,
                      filter_kwargs={},which_chi = 'src',solver_type='PROFILE_SHEAR'):

        # uses source plane chi^2

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']
        assert which_chi is not None
        if which_chi == 'src':
            basic_or_full = 'basic'
        else:
            basic_or_full = 'full'

        lens_systems= []

        if realizations is not None:
            for real in realizations:
                lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),additional_halos=real,multiplane=multiplane,
                                                      filter_by_position=filter_by_position,**filter_kwargs))
        else:
            lens_systems.append(self.build_system(main=copy.deepcopy(macromodel),multiplane=multiplane))

        optimized_data, model = self.optimize_4imgs(lens_systems=lens_systems, data2fit=datatofit, method=method,
                                                    sigmas=sigmas, identifier=identifier, grid_rmax=grid_rmax,
                                                    res=res, source_shape=source_shape, ray_trace=ray_trace,
                                                    source_size=source_size, print_mag=print_mag, opt_routine=basic_or_full,
                                                    raytrace_with=raytrace_with, polar_grid=polar_grid,solver_type=solver_type)

        return optimized_data,model

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

        magnifications, images = trace.get_images(xpos=[xcoord], ypos=[ycoord], lens_system=lens_systems[0])

        return magnifications, images










