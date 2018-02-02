from MagniPy.magnipy import Magnipy
from RayTrace.raytrace import RayTrace

class SolveRoutines(Magnipy):
    """
    This class uses the routines set up in MagniPy to solve the lens equation in various ways with lenstronomy or lensmodel
    """

    def solve_lens_equation(self,full_system=None,macromodel=None,realizations=None,multiplane=None,method=None,ray_trace=None, sigmas=None,
                             identifier=None,srcx=None, srcy=None, gridsize=None,res=None,
                             source_shape='GAUSSIAN', source_size=None, print_mag=False):

        self.reset()

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                for real in realizations:
                    self.build_system(main=macromodel, additional_halos=real, multiplane=multiplane)

        else:
            assert macromodel is None

            self.build_system(main=full_system.lens_components[0],additional_halos=full_system.lens_components[1:])


        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']



        data = self.solve_4imgs(method=method,sigmas=sigmas,identifier=identifier,srcx=srcx,srcy=srcy,gridsize=gridsize,
                                res=res,source_shape=source_shape,ray_trace=ray_trace,source_size=source_size)
        self.reset()

        return data


    def two_step_optimize(self,macromodel,datatofit,realizations,multiplane,method=None,ray_trace=None, sigmas=None,
                             identifier=None,srcx=None, srcy=None, gridsize=None,res=None,
                             source_shape='GAUSSIAN', source_size=None, print_mag=False):

        # optimizes the macromodel first, then uses it to optimize with additional halos in the lens model
        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        _,macromodel = self.macromodel_initialize(macromodel=macromodel,datatofit=datatofit,realizations=realizations,
                                                    multiplane=multiplane,method=method,ray_trace=ray_trace,sigmas=sigmas,
                                                    identifier=identifier,srcx=srcx,srcy=srcy,gridsize=gridsize,res=res,
                                                    source_shape=source_shape,source_size=source_size,print_mag=print_mag)


        optimized_data, newsystem = self.fit_src_plane(macromodel=macromodel, datatofit=datatofit, realizations=realizations, multiplane=multiplane, method=method,
                                                       ray_trace=ray_trace, sigmas=sigmas, identifier=identifier, srcx=srcx, srcy=srcy, gridsize=gridsize, res=res,
                                                       source_shape=source_shape, source_size=source_size, print_mag=print_mag)

        return optimized_data,newsystem

    def macromodel_initialize(self,macromodel,datatofit,realizations,multiplane,method=None,ray_trace=None, sigmas=None,
                             identifier=None,srcx=None, srcy=None, gridsize=None,res=None,
                             source_shape='GAUSSIAN', source_size=None, print_mag=False):

        # fits just a single macromodel profile to the data

        assert method is not None
        assert method in ['lensmodel','lenstronomy']

        self.reset()

        self.build_system(main=macromodel, additional_halos=None, multiplane=multiplane)

        optimized_data, model = self.optimize_4imgs(data2fit=datatofit, method=method,
                                                         sigmas=sigmas, identifier=identifier, gridsize=gridsize,
                                                         res=res, source_shape=source_shape, ray_trace=False,
                                                         source_size=source_size, print_mag=print_mag, opt_routine='randomize')
        macromodel = model[0].lens_components[0]
        self.reset()

        return optimized_data,macromodel

    def fit_src_plane(self, macromodel, datatofit, realizations, multiplane, method=None, ray_trace=None, sigmas=None,
                      identifier=None, srcx=None, srcy=None, gridsize=None, res=None,
                      source_shape='GAUSSIAN', source_size=None, print_mag=False):

        # uses source plane chi^2

        assert method is not None
        assert method in ['lensmodel', 'lenstronomy']

        self.reset()

        if realizations is not None:
            for real in realizations:
                self.build_system(main=macromodel,additional_halos=real,multiplane=multiplane)
        else:
            self.build_system(main=macromodel,multiplane=multiplane)

        optimized_data, model = self.optimize_4imgs(data2fit=datatofit, method=method,
                                          sigmas=sigmas, identifier=identifier, gridsize=gridsize,
                                          res=res, source_shape=source_shape, ray_trace=ray_trace,
                                          source_size=source_size, print_mag=print_mag, opt_routine='basic')
        lenssystems = self.lens_systems

        self.reset()

        return optimized_data,lenssystems

    def fit_imgplane(self, macromodel, datatofit, realizations, multiplane, method=None, ray_trace=None,
                         sigmas=None,
                         identifier=None, srcx=None, srcy=None, gridsize=None, res=None,
                         source_shape='GAUSSIAN', source_size=None, print_mag=False):

        # uses image plane chi^2; quite slow

        assert method is not None
        assert method in ['lensmodel']

        for real in realizations:
            self.build_system(main=macromodel, additional_halos=real)

        optimized_data, model = self.optimize_4imgs(data2fit=datatofit, method=method,
                                                       sigmas=sigmas, identifier=identifier, gridsize=gridsize,
                                                       res=res, source_shape=source_shape, ray_trace=ray_trace,
                                                       source_size=source_size, print_mag=print_mag,
                                                       opt_routine='full')
        self.reset()

        return optimized_data, self.lens_systems

    def produce_images(self, full_system=None, macromodel=None, realizations=None, multiplane=None, method=None,
                       identifier=None, srcx=None, srcy=None, gridsize=None, res=None,
                       source_shape='GAUSSIAN', source_size=None):

        self.reset()

        if full_system is None:
            assert macromodel is not None
            if realizations is not None:
                assert len(realizations) == 1
                for real in realizations:
                    self.build_system(main=macromodel, additional_halos=real, multiplane=multiplane)

        else:
            assert macromodel is None

            self.build_system(main=full_system.lens_components[0], additional_halos=full_system.lens_components[1:])

        data = self.solve_lens_equation(full_system=self.lens_systems[0], multiplane=multiplane, method=method,
                                        identifier=None, srcx=None, srcy=None,
                                        gridsize=None, res=None, source_shape='GAUSSIAN', source_size=None)

        trace = RayTrace(xsrc=srcx, ysrc=srcy, multiplane=multiplane, method=method, gridsize=gridsize, res=res,
                         source_shape=source_shape,
                         cosmology=self.cosmo.cosmo, source_size=source_size)

        magnifications, images = trace.get_images(xpos=data[0].x, ypos=data[0].y, lens_system=self.lens_systems[0])

        self.reset()

        return magnifications, images










