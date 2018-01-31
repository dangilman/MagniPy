import LensBuild.lens_assemble as build
import LensBuild.renderhalos as halo_gen
from LensBuild.spatial_distribution import *
from LensBuild.cosmology import Cosmo
from MagniPy.Solver.GravlensWrap._call import *
from MagniPy.Solver.GravlensWrap.generate_input import *
from MagniPy.Solver.LenstronomyWrap.generate_input import *
from MagniPy.Solver.RayTrace import raytrace
from MagniPy.lensdata import Data
from Solver.GravlensWrap.gravlens_to_kwargs import translate
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import time


class Magnipy:

    """
    This class is used to specify a lens system and solve the lens equation or optimize a lens model

    """

    def __init__(self,zmain,zsrc,main_lens_profile=[],use_lenstronomy_halos=False):

        self.zmain = zmain
        self.zsrc = zsrc
        self.lens_halos = halo_gen.HaloGen(z_l=zmain, z_s=zsrc, cosmo='FlatLambdaCDM',use_lenstronomy_halos=use_lenstronomy_halos)
        self.lens_systems = []
        self.cosmo = Cosmo(zd=zmain,zsrc=zsrc)

    def print_system(self,system_index,component_index=None):

        system = self.lens_systems[system_index]
        if component_index is not None:
            system.lens_components[component_index].print_args()
        else:
            for component in system.lens_components:
                component.print_args()

    def reset(self):

        self.lens_systems = []
        self.halos = None
        self.main = None

    def update_system(self,system_index,component_index,newkwargs,method):

        self.lens_systems[system_index].lens_components[component_index].update(method=method,**newkwargs)

    def build_system(self,main=None,additional_halos=None,multiplane=None):

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

        self.lens_systems.append(newsystem)

    def _spatial_halo_filter(self, halos, xpos=None, ypos=None, mindis=None, masscut_low=None):

        newhalos = filter_spatial(halos, xpos=xpos, ypos=ypos, mindis=mindis, masscut_low=masscut_low)

        return newhalos

    def generate_halos(self,halomodel,Nrealizations=1,spatial_filter=False,**spatial_kwargs):

        self.lens_halos.substructure_init(halomodel)
        realizations = self.lens_halos.draw_subhalos(N_real=Nrealizations)

        if spatial_filter:
            newrealizations = []
            for realization in realizations:
                newrealizations.append(self._spatial_halo_filter(realization, **spatial_kwargs))
            return newrealizations
        else:
            return realizations


    def optimize_4imgs(self, data2fit=[], method=str, sigmas=[], identifier='', opt_routine=None,
                       ray_trace = True, return_positions = False, gridsize=int, res=0.0005, source_shape='GAUSSIAN',
                       source_size=float, print_mag=False):

        # opt_routine:
        # basic: gridflag = 0, chimode = 0; optimizes in source plane, fast
        # full: gridflag = 1, chimode = 1; optimizes in image plane, flow
        # randomize: does a randomize command command followed by full

        d2fit = [data2fit.x, data2fit.y, data2fit.m, data2fit.t]

        if method == 'lensmodel':

            assert opt_routine is not None
            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            macroindex = []
            for system in self.lens_systems:
                full = FullModel(multiplane=system.multiplane)
                for i, model in enumerate(system.lens_components):
                    if model.tovary:
                        macroindex.append(i)
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=d2fit, zlens=self.zmain, zsrc=self.zsrc, opt_routine=opt_routine)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lensdata,macromodels = [],[]

            for i,name in enumerate(outputfile):

                xvals, yvals, mag_gravlens, tvals, macrovals, srcvals = read_dat_file(fname=name)

                lensdata.append(Data(x=xvals, y=yvals, m=mag_gravlens, t=tvals, source=srcvals))

                newmacromodel = translate(macrovals)

                macromodels.append(newmacromodel)

                self.update_system(system_index=i, component_index=0, newkwargs=newmacromodel, method='lensmodel')

            t0 = time.time()

            if ray_trace:

                for i, name in enumerate(outputfile):

                    if print_mag:
                        print 'computing mag #: ',i+1

                    ray_shooter = raytrace.RayTrace(xsrc=lensdata[i].srcx,ysrc=lensdata[i].srcy,gridsize=gridsize,
                                                    source_size=source_size,res=res,source_shape=source_shape,
                                                    cosmology=self.cosmo.cosmo,zsrc=self.zsrc)

                    magnification = ray_shooter.compute_mag(xpos=lensdata[i].x,ypos=lensdata[i].y,
                                                            lens_system=self.lens_systems[i],print_mag=False)

                    lensdata[i].set_mag(magnification)

                    lensdata[i].sort_by_pos(data2fit.x, data2fit.y)


            return lensdata, self.lens_systems,(time.time() - t0)

        elif method == 'lenstronomy':

            data = []

            T = 0

            for i, system in enumerate(self.lens_systems):

                lenstronomywrap = LenstronomyWrap(multiplane=system.multiplane,cosmo=self.cosmo.cosmo, z_source=self.zsrc)

                lenstronomywrap.assemble(system)

                #print self.lens_systems[0].lens_components[0].lenstronomy_args

                kwargs_fit = lenstronomywrap.optimize_lensmodel(d2fit[0], d2fit[1])

                #print self.lens_systems[0].lens_components[0].lenstronomy_args
                #exit(1)

                self.update_system(system_index=i, component_index=0, newkwargs=kwargs_fit,method='lenstronomy')

                lenstronomywrap.update_lensparams(component_index=0, newkwargs=kwargs_fit)

                lensModel = lenstronomywrap.model

                xsrc, ysrc = lensModel.ray_shooting(d2fit[0], d2fit[1], lenstronomywrap.lens_model_params)

                x_image, y_image = lenstronomywrap.solve_leq(xsrc=np.mean(xsrc),ysrc=np.mean(ysrc))


                newdata = Data(x=x_image, y=y_image, m=None, t=None, source=[xsrc, ysrc])

                t0 = time.time()

                if ray_trace:

                    if print_mag:
                        print 'computing mag #: ',i+1

                    lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list,
                                                              z_source=self.zsrc, redshift_list=system.redshift_list,
                                                              cosmo=self.cosmo.cosmo, multi_plane=system.multiplane)

                    fluxes = lensModelExtensions.magnification_finite(x_pos=x_image,
                                                                      y_pos=y_image,
                                                                      kwargs_lens=lenstronomywrap.lens_model_params,
                                                                      source_sigma=source_size, window_size=gridsize,
                                                                      grid_number=gridsize * res ** -1,
                                                                      shape=source_shape)

                    newdata.set_mag(fluxes)

                    newdata.sort_by_pos(data2fit.x, data2fit.y)

                    T += time.time() - t0

                data.append(newdata)

            return data, self.lens_systems, T

    def solve_4imgs(self, method=str, sigmas=[], identifier='', srcx=None, srcy=None, gridsize=.1,
                    res=0.001, source_shape='GAUSSIAN', ray_trace=True, source_size=float, print_mag=False,time_ray_trace=False):


        if method == 'lensmodel':

            data=[]

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            for i,system in enumerate(self.lens_systems):
                full = FullModel(multiplane=system.multiplane)
                for model in system.lens_components:
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=None, zlens=self.zmain, zsrc=self.zsrc,srcx=srcx,srcy=srcy)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lens_data = read_gravlens_out(fnames=outputfile)

            t0 = time.time()

            for i,system in enumerate(self.lens_systems):
                x, y, m, t, nimg = lens_data[i]
                data.append(Data(x=x,y=y,m=m,t=t, source=[srcx, srcy]))

                if ray_trace:

                    if print_mag:
                        print 'computing mag #: ',i+1

                    ray_shooter = raytrace.RayTrace(xsrc=srcx, ysrc=srcy, multiplane=self.lens_systems[i].multiplane,
                                                    gridsize=gridsize,res=res,source_shape=source_shape,
                                                    source_size=source_size,cosmology = self.cosmo.cosmo,zsrc=self.zsrc)

                    fluxes = ray_shooter.compute_mag(data[i].x,data[i].y,self.lens_systems[i])

                    data[i].set_mag(fluxes)

                    data[i].sort_by_pos(x, y)

            return data,(time.time()-t0)

        elif method == 'lenstronomy':

            data = []

            for i, system in enumerate(self.lens_systems):

                lenstronomywrap = LenstronomyWrap(multiplane=system.multiplane, cosmo=self.cosmo.cosmo,
                                                  z_source=self.zsrc)

                lenstronomywrap.assemble(system)

                x_image,y_image = lenstronomywrap.solve_leq(xsrc=srcx,ysrc=srcy)

                data.append(Data(x=x_image, y=y_image, m=None, t=None, source=[srcx, srcy]))

            t0 = time.time()

            if raytrace:
                for i, system in enumerate(self.lens_systems):

                    if print_mag:
                        print 'computing mag #: ',i+1

                    lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list,
                                                              z_source=self.zsrc, redshift_list=system.redshift_list,
                                                              cosmo=self.cosmo.cosmo,multi_plane=system.multiplane)

                    fluxes = lensModelExtensions.magnification_finite(x_pos=data[i].x,
                                                                      y_pos=data[i].y,
                                                                      kwargs_lens=lenstronomywrap.lens_model_params,
                                                                      source_sigma=source_size, window_size=gridsize,
                                                                      grid_number=gridsize * res ** -1,
                                                                      shape=source_shape)
                    data[i].set_mag(fluxes)

                    data[i].sort_by_pos(data[i].x, data[i].y)


            return data,(time.time()-t0)












