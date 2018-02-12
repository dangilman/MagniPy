import time
import MagniPy.LensBuild.lens_assemble as build
import MagniPy.LensBuild.renderhalos as halo_gen
from MagniPy.LensBuild.cosmology import Cosmo
from MagniPy.Solver.GravlensWrap._call import *
from MagniPy.Solver.GravlensWrap.generate_input import *
from MagniPy.Solver.GravlensWrap.gravlens_to_kwargs import gravlens_to_kwargs
from MagniPy.Solver.LenstronomyWrap.generate_input import *
from MagniPy.Solver.RayTrace import raytrace
from MagniPy.lensdata import Data
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from MagniPy.util import filter_by_position


class Magnipy:

    """
    This class is used to specify a lens system and solve the lens equation or optimize a lens model

    """

    def __init__(self,zmain,zsrc,main_lens_profile=[],use_lenstronomy_halos=False):

        self.zmain = zmain
        self.zsrc = zsrc
        self.lens_halos = halo_gen.HaloGen(z_l=zmain, z_s=zsrc, cosmo='FlatLambdaCDM',use_lenstronomy_halos=use_lenstronomy_halos)

        self.cosmo = Cosmo(zd=zmain,zsrc=zsrc)

    def print_system(self,system_index,component_index=None):

        system = self.lens_system[system_index]
        if component_index is not None:
            system.lens_components[component_index].print_args()
        else:
            for component in system.lens_components:
                component.print_args()

    def update_system(self,lens_system,component_index,newkwargs,method):

        lens_system.lens_components[component_index].update(method=method, **newkwargs)

        return lens_system

    def build_system(self,main=None,additional_halos=None,multiplane=None,filter_by_pos=False,**filter_kwargs):

        """
        This routine sets up the lens model. It assembles a "system", which is a list of Deflector instances
        :param main: instance of Deflector for the main deflector
        :param halo_model: specifies the halo content of the deflector
        :param realization: specifies a specific realization; suborinate to halo_model
        :param multiplane: whether halo_model includes multiple lens planes
        :return:
        """

        assert multiplane is not None

        newsystem = build.LensSystem(multiplane=multiplane)

        newsystem.main_lens(main)

        if additional_halos is not None:
            newsystem.halos(additional_halos)

        if filter_by_pos:

            newsystem.filter_by_position(**filter_kwargs)

        return newsystem

    def generate_halos(self,halomodel,Nrealizations=1,filter_by_pos=False,**spatial_kwargs):

        self.lens_halos.substructure_init(halomodel)
        realizations = self.lens_halos.draw_subhalos(N_real=Nrealizations)

        if filter_by_pos:
            _realizations = []
            for realization in realizations:
                newrealization,newredshift = filter_by_position(realization,**spatial_kwargs)
                _realizations.append(newrealization)
            return _realizations
        else:
            return realizations

    def optimize_4imgs(self, lens_systems=None, data2fit=[], method=str, sigmas=[], identifier='', opt_routine=None,
                       ray_trace = True, return_positions = False, grid_rmax=int, res=0.0005, source_shape='GAUSSIAN',
                       source_size=float, print_mag=False, raytrace_with=None, polar_grid=False):

        # opt_routine:
        # basic: gridflag = 0, chimode = 0; optimizes in source plane, fast
        # full: gridflag = 1, chimode = 1; optimizes in image plane, flow
        # randomize: does a randomize command command followed by full

        d2fit = [data2fit.x, data2fit.y, data2fit.m, data2fit.t]

        optimized_systems = []

        if method == 'lensmodel':

            assert opt_routine is not None
            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            for system in lens_systems:

                full = FullModel(multiplane=system.multiplane)
                for i, model in enumerate(system.lens_components):

                    full.populate(SingleModel(lensmodel=model))

                solver.add_lens_system(full)

            outputfile = solver.write_all(data=d2fit, zlens=self.zmain, zsrc=self.zsrc, opt_routine=opt_routine)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lensdata = []

            for i,name in enumerate(outputfile):

                xvals, yvals, mag_gravlens, tvals, macrovals, srcvals = read_dat_file(fname=name)

                lensdata.append(Data(x=xvals, y=yvals, m=mag_gravlens, t=tvals, source=srcvals))

                #newmacromodel = translate(macrovals,vary_shear=lens_systems)

                newmacromodel = gravlens_to_kwargs(macrovals, deflector = lens_systems[i].lens_components[0])

                optimized_systems.append(self.update_system(lens_system=lens_systems[i], component_index=0, newkwargs=newmacromodel, method='lensmodel'))

            if ray_trace:

                for i, name in enumerate(outputfile):

                    if print_mag:
                        print 'computing mag #: ', i + 1

                    fluxes = self.do_raytrace_lensmodel(lens_system=optimized_systems[i], xpos=lensdata[i].x,
                                                        ypos=lensdata[i].y,
                                                        xsrc=lensdata[i].srcx, ysrc=lensdata[i].srcy,
                                                        multiplane=optimized_systems[i].multiplane,
                                                        grid_rmax=grid_rmax,
                                                        res=res, source_shape=source_shape, source_size=source_size,
                                                        cosmology=self.cosmo, zsrc=self.zsrc, raytrace_with = raytrace_with, polar_grid=polar_grid)

                    lensdata[i].set_mag(fluxes)


            for dataset in lensdata:
                dataset.sort_by_pos(data2fit.x,data2fit.y)

            return lensdata, optimized_systems

        elif method == 'lenstronomy':

            data = []

            optimized_systems = []

            for i, system in enumerate(lens_systems):

                lenstronomywrap = LenstronomyWrap(multiplane=system.multiplane,cosmo=self.cosmo.cosmo, z_source=self.zsrc)

                lenstronomywrap.assemble(system)

                #print lens_systems[0].lens_components[0].lenstronomy_args

                kwargs_fit = lenstronomywrap.optimize_lensmodel(d2fit[0], d2fit[1])

                #print lens_systems[0].lens_components[0].lenstronomy_args
                #exit(1)

                optimized_systems.append(self.update_system(lens_system=lens_systems[i],component_index=0,
                                                            newkwargs=kwargs_fit,method='lenstronomy'))

                lenstronomywrap.update_lensparams(component_index=0, newkwargs=kwargs_fit)

                lensModel = lenstronomywrap.model

                xsrc, ysrc = lensModel.ray_shooting(d2fit[0], d2fit[1], lenstronomywrap.lens_model_params)
                #xsrc,ysrc = -0.0093,0.0023
                x_image, y_image = lenstronomywrap.solve_leq(xsrc=np.mean(xsrc),ysrc=np.mean(ysrc))

                newdata = Data(x=x_image, y=y_image, m=None, t=None, source=[xsrc, ysrc])

                t0 = time.time()

                if ray_trace:

                    if print_mag:
                        print 'computing mag #: ',i+1

                    fluxes = self.do_raytrace_lenstronomy(lenstronomy_wrap_instance=lenstronomywrap, xpos=newdata.x,
                                                          ypos=newdata.y, source_size=source_size, gridsize=grid_rmax, res=res,
                                                          source_shape=source_shape, zsrc=self.zsrc, cosmology=self.cosmo.cosmo,
                                                          multiplane=system.multiplane)

                    newdata.set_mag(fluxes)

                data.append(newdata)

            for dataset in data:
                dataset.sort_by_pos(data2fit.x,data2fit.y)

            return data, optimized_systems

    def solve_4imgs(self, lens_systems = None, method=str, sigmas=[], identifier='', srcx=None, srcy=None, grid_rmax=.1,
                    res=0.001, source_shape='GAUSSIAN', ray_trace=True, source_size=float, print_mag=False, time_ray_trace=False):


        if method == 'lensmodel':

            data=[]

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            for i,system in enumerate(lens_systems):
                full = FullModel(multiplane=system.multiplane)
                for model in system.lens_components:
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=None, zlens=self.zmain, zsrc=self.zsrc,srcx=srcx,srcy=srcy)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lens_data = read_gravlens_out(fnames=outputfile)
            t0 = time.time()

            for i,system in enumerate(lens_systems):

                x, y, m, t, nimg = lens_data[i]

                data.append(Data(x=x,y=y,m=m,t=t, source=[srcx, srcy]))

                if ray_trace:

                    if print_mag:
                        print 'computing mag #: ',i+1

                    fluxes = self.do_raytrace_lensmodel(lens_system=system, xpos=data[i].x, ypos=data[i].y, xsrc=srcx,
                                                        ysrc=srcy, multiplane=lens_systems[i].multiplane, grid_rmax=grid_rmax,
                                                        res=res, source_shape=source_shape, source_size=source_size,
                                                        cosmology = self.cosmo, zsrc=self.zsrc)

                    data[i].set_mag(fluxes)

            if ray_trace:
                print 'time to ray trace (min): ', np.round((time.time() - t0) * 60 ** -1, 1)

            return data

        elif method == 'lenstronomy':

            data = []

            for i, system in enumerate(lens_systems):

                lenstronomywrap = LenstronomyWrap(multiplane=system.multiplane, cosmo=self.cosmo.cosmo,
                                                  z_source=self.zsrc)

                lenstronomywrap.assemble(system)

                x_image,y_image = lenstronomywrap.solve_leq(xsrc=srcx,ysrc=srcy)

                data.append(Data(x=x_image, y=y_image, m=None, t=None, source=[srcx, srcy]))

            if raytrace:
                t0 = time.time()
                for i, system in enumerate(lens_systems):

                    if print_mag:
                        print 'computing mag #: ',i+1

                    fluxes = self.do_raytrace_lenstronomy(lenstronomy_wrap_instance=lenstronomywrap, xpos=data[i].x, ypos=data[i].y,
                                                          source_size=source_size, gridsize=grid_rmax,
                                                          res=res, source_shape=source_shape, zsrc=self.zsrc,
                                                          cosmology=self.cosmo.cosmo, multiplane=system.multiplane)

                    data[i].set_mag(fluxes)

            if ray_trace:
                print 'time to ray trace (min): ', np.round((time.time() - t0) * 60 ** -1, 1)

            return data

    def do_raytrace_lensmodel(self, lens_system, xpos, ypos, xsrc=float, ysrc=float, multiplane=None, grid_rmax=None,
                              res=None, source_shape=None, source_size=None, cosmology = classmethod, zsrc=None, raytrace_with=None,
                              polar_grid=False):

        ray_shooter = raytrace.RayTrace(xsrc=xsrc, ysrc=ysrc, multiplane=multiplane,
                                        grid_rmax=grid_rmax, res=res, source_shape=source_shape,
                                        source_size=source_size, cosmology=cosmology, zsrc=self.zsrc,
                                        raytrace_with=raytrace_with, polar_grid=polar_grid)

        fluxes = ray_shooter.compute_mag(xpos,ypos,lens_system=lens_system)

        return fluxes

    def do_raytrace_lenstronomy(self, lenstronomy_wrap_instance, xpos, ypos, multiplane=None, gridsize=None,
                                res=None, source_shape=None, source_size=None, cosmology=classmethod, zsrc=None):



        lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomy_wrap_instance.lens_model_list,
                                                  z_source=zsrc, redshift_list=lenstronomy_wrap_instance.redshift_list,
                                                  cosmo=cosmology, multi_plane=multiplane)

        fluxes = lensModelExtensions.magnification_finite(x_pos=xpos,
                                                          y_pos=ypos,
                                                          kwargs_lens=lenstronomy_wrap_instance.lens_model_params,
                                                          source_sigma=source_size, window_size=gridsize,
                                                          grid_number=gridsize * res ** -1,
                                                          shape=source_shape)


        return fluxes














