import LensBuild.lens_assemble as build
import LensBuild.renderhalos as halo_gen
from Solver.GravlensWrap.gravlens_to_kwargs import translate
from Solver.lensdata import LensData
from MagniPy.util import *
from MagniPy.Solver.GravlensWrap.generate_input import *
from MagniPy.Solver.GravlensWrap._call import *
from MagniPy.Solver.LenstronomyWrap.generate_input import *
from MagniPy.Solver.RayTrace import raytrace

class Magnipy:

    """
    This class is used to specify a lens system and solve the lens equation or optimize a lens model

    """

    def __init__(self,zmain,zsrc,main_lens_profile=[]):

        self.zmain = zmain
        self.zsrc = zsrc
        self.lens_halos = halo_gen.HaloGen(z_l=zmain, z_s=zsrc, cosmo='FlatLambdaCDM')
        self.lens_systems = []

    def print_system(self,system_index):

        system = self.lens_systems[system_index]

        for component in system.lens_components:
            component.print_args()

    def reset(self):

        self.lens_systems = []

    def update_system(self,system_index,component_index,newkwargs):

        self.lens_systems[system_index].lens_components[component_index].update(**newkwargs)

    def build_system(self,main=None,halo_model=None,realization=None,multiplane=None):

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

        self._set_main(main_model=None,main=main)
        self._set_halos(halo_model,realization)

        newsystem.main_lens(self.main)
        if self.halos is not None:
            newsystem.halos(self.halos)
        self.lens_systems.append(newsystem)


    def _set_main(self,main_model=None,main=None):

        self.main_set = True
        self.main = main

    def _set_halos(self,halo_model=None,realization=None):

        if halo_model is None:
            self.halos_set = True
            self.halos = realization
        elif realization is not None:
            self.halos_set = True
            self.lens_halos.substructure_init(model=halo_model)
            self.halos = self.lens_halos.draw_subhalos(N_real=1)[0]

    def generate_halos(self,halomodel,Nrealizations=1):

        self.lens_halos.substructure_init(halomodel)
        return self.lens_halos.draw_subhalos(N_real=Nrealizations)

    def optimize_4imgs(self, data2fit=[], method=str, sigmas=[], identifier='', opt_routine=None,
                       return_fluxes = True, return_positions = False,source_sigma=float,gridsize=int,res=0.001,source_shape='GAUSSIAN',
                       source_size=float):

        # opt_routine:
        # basic: gridflag = 0, chimode = 0; optimizes in source plane, fast
        # full: gridflag = 1, chimode = 1; optimizes in image plane, flow
        # randomize: does a randomize command command followed by full
        assert opt_routine is not None

        if method == 'lensmodel':

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

            outputfile = solver.write_all(data=data2fit, zlens=self.zmain, zsrc=self.zsrc, opt_routine=opt_routine)

            call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lensdata = []
            macromodels = []

            for i, name in enumerate(outputfile):

                xvals, yvals, mag_gravlens, tvals, macrovals, srcvals = read_dat_file(fname=name)

                data = LensData(x=xvals, y=yvals, m=mag_gravlens, t=tvals, source=srcvals)

                newmacromodel = translate(macrovals)

                macromodels.append(newmacromodel)

                self.update_system(system_index=i, component_index=0, newkwargs=newmacromodel)

                if return_fluxes:

                    ray_shooter = raytrace.RayTrace(xsrc=srcvals[0], ysrc=srcvals[0], multiplane=self.lens_systems.multiplane,
                                                    gridsize=gridsize,res=res,source_shape=source_shape,source_size=source_size)
                    data.m = ray_shooter.compute_mag(data.x,data.y,self.lens_systems[i])

                lensdata.append(data)

            return self.lens_systems,lensdata

        elif method == 'lenstronomy':

            lenstronomywrap = LenstronomyWrap()

            data = []

            for i, system in enumerate(self.lens_systems):

                lenstronomywrap.assemble(system, include_shear=False)

                kwargs_fit = lenstronomywrap.optimize_lensmodel(data2fit[0], data2fit[1])

                self.update_system(system_index=i, component_index=0, newkwargs=kwargs_fit)

                lenstronomywrap.update_lensparams(system_index=0, newkwargs=kwargs_fit)

                lensModel = LensModel(lenstronomywrap.lens_model_list)

                xsrc,ysrc = lensModel.ray_shooting(x_image, y_image, lenstronomywrap.lens_model_params)

                x_image, y_image = lenstronomywrap.solve_leq(xsrc=xsrc,ysrc=ysrc)

                newdata = LensData(x=x_image, y=y_image, m=None, t=None, source=[xsrc, ysrc])

                if return_fluxes:

                    lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list)
                    fluxes = lensModelExtensions.magnification_finite(x_pos = x_image,
                                                                      y_pos=y_image,kwargs_lens=lenstronomywrap.lens_model_params,
                                                                      source_sigma=source_sigma,window_size=gridsize,
                                                                      grid_number=gridsize*res**-1,shape=source_shape)
                    newdata.m = fluxes

                data.append(newdata)

            return self.lens_systems,data


    def solve_4imgs(self, method=str, sigmas=[], identifier='',srcx=None,srcy=None,source_sigma=float,gridsize=int,
                    res=0.001,source_shape='GAUSSIAN',return_fluxes=True,source_size=float):

        if method == 'lensmodel':

            data=[]

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            for i,system in enumerate(self.lens_systems):
                full = FullModel(multiplane=system.multiplane)
                for i, model in enumerate(system.lens_components):
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

                outputfile = solver.write_all(data=None, zlens=self.zmain, zsrc=self.zsrc,srcx=srcx,srcy=srcy)

                call_lensmodel(inputfile=solver.outfile_path + solver.filename + '.txt',
                               path_2_lensmodel=path_2_lensmodel)

                x,y,m,t,nimg = read_gravlens_out(fnames=outputfile)[0]
                lensdata = LensData(x=x, y=y, m=m, t=t, source=[srcx,srcy])

                if return_fluxes:

                    ray_shooter = raytrace.RayTrace(xsrc=srcx, ysrc=srcy, multiplane=self.lens_systems.multiplane,
                                                    gridsize=gridsize,res=res,source_shape=source_shape,source_size=source_size)
                    lensdata.m = ray_shooter.compute_mag(data.x,data.y,self.lens_systems[i])

                data.append(lensdata)

            return data

        elif method == 'lenstronomy':

            lenstronomywrap = LenstronomyWrap()

            data = []

            for i, system in enumerate(self.lens_systems):
                lenstronomywrap.assemble(system, include_shear=False)

                x_image,y_image = lenstronomywrap.solve_leq(xsrc=srcx,ysrc=srcy)

                newdata = LensData(x=x_image, y=y_image, m=None, t=None, source=[srcx, srcy])

                if return_fluxes:

                    lensModelExtensions = LensModelExtensions(lens_model_list=lenstronomywrap.lens_model_list)
                    fluxes = lensModelExtensions.magnification_finite(x_pos=x_image,
                                                                      y_pos=y_image,
                                                                      kwargs_lens=lenstronomywrap.lens_model_params,
                                                                      source_sigma=source_sigma, window_size=gridsize,
                                                                      grid_number=gridsize * res ** -1,
                                                                      shape=source_shape)
                    newdata.m = fluxes

                data.append(newdata)

            return data












