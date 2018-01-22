import LensBuild.lens_assemble as build
import LensBuild.renderhalos as halo_gen
from Solver.GravlensWrap.generate_input import *
from Solver.GravlensWrap.gravlens_to_kwargs import translate
from Solver.SolveLEQ import *
from Solver.lensdata import LensData

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


    def optimize_macro_4imgs(self,data2fit=[],method=str,sigmas=[],identifier=''):

        # opt_routine:
        # basic: gridflag = 0, chimode = 0; optimizes in source plane, fast
        # full: gridflag = 1, chimode = 1; optimizes in image plane, flow
        # randomize: does a randomize command command followed by full

        if method == 'lensmodel':

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                           pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                           identifier=identifier)

            for system in self.lens_systems:
                full = FullModel(multiplane=system.multiplane)
                for i, model in enumerate(system.lens_components):
                    if model.tovary:
                        macroindex = i
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=data2fit,zlens=self.zmain, zsrc=self.zsrc, opt_routine='randomize')

            lensmodel_call(inputfile=solver.outfile_path+solver.filename+'.txt',path_2_lensmodel=path_2_lensmodel)

            x,y,m,t,macro,src = read_dat_file(fname=outputfile[0])

            data = LensData(x=x, y=y, m=m, t=t, source=src)

            newmacromodel = translate(macro)

            self.update_system(system_index=0,component_index=0,newkwargs=newmacromodel)

            return self.lens_systems[0].lens_components[0],data

    def optimize_4imgs(self, data2fit=[], method=str, sigmas=[], identifier='', opt_routine=None):

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

            lensmodel_call(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            lensdata = []
            macromodels = []

            for name in outputfile:
                xvals, yvals, mvals, tvals, macrovals, srcvals = read_dat_file(fname=name)

                data = LensData(x=xvals, y=yvals, m=mvals, t=tvals, source=srcvals)
                lensdata.append(data)

                newmacromodel = translate(macrovals)

                macromodels.append(newmacromodel)

            for i,system in enumerate(self.lens_systems):
                system.lens_components[macroindex[i]].update(**macromodels[i])

            return self.lens_systems,lensdata


    def solve_4imgs(self, data2fit=[], method=str, sigmas=[], identifier='',srcx=None,srcy=None):

        if method == 'lensmodel':

            solver = GravlensInput(filename=identifier, zlens=self.zmain, zsrc=self.zsrc,
                                   pos_sigma=sigmas[0], flux_sigma=sigmas[1], tdelay_sigma=sigmas[2],
                                   identifier=identifier)

            for system in self.lens_systems:
                full = FullModel(multiplane=system.multiplane)
                for i, model in enumerate(system.lens_components):
                    full.populate(SingleModel(lensmodel=model))
                solver.add_lens_system(full)

            outputfile = solver.write_all(data=None, zlens=self.zmain, zsrc=self.zsrc,srcx=srcx,srcy=srcy)

            lensmodel_call(inputfile=solver.outfile_path + solver.filename + '.txt',
                           path_2_lensmodel=path_2_lensmodel)

            x,y,m,t,nimg = read_gravlens_out(fnames=outputfile)[0]
            data = LensData(x=x, y=y, m=m, t=t, source=[None,None])
            return data

        else:
            raise ValueError('lenstronomy not yet implemented')












