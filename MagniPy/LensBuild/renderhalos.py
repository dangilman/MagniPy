from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from spatial_distribution import *
from massfunctions import *
from MagniPy.MassModels import TNFW
from MagniPy.MassModels import NFW
from MagniPy.MassModels import PJaffe
from MagniPy.MassModels import PointMass
from MagniPy.LensBuild.lens_assemble import Deflector


class HaloGen:

    def __init__(self,z_l=0.5,z_s=1.5,cosmo='FlatLambdaCDM',use_lenstronomy_halos=False):

        self.cosmology = Cosmo(zd=z_l,zsrc=z_s,cosmology=cosmo)

        self.substrucutre_initialized = False

        self.realizations = []

        self.use_lenstronomy_halos = use_lenstronomy_halos

        self.zd,self.zsrc = self.cosmology.zd,self.cosmology.zsrc

        #if self.use_lenstronomy_halos:
        #    from lenstronomy.LensModel.Profiles import *

    def add_single_plane(self, zplane, model, sub_mod_index, dz=float):

        subhalos =[]

        spatial = model['spatial'][sub_mod_index][0]
        assert type(spatial) is str

        if spatial == 'uniformflat':
            Rmax_2d = model['spatial'][sub_mod_index][1][0]
            trunc_rad = model['spatial'][sub_mod_index][1][1]
            geometric_factor = self.cosmology.D_ratio([0,zplane],[0,self.zd])
            geometric_factor = 1
            self.spatial = Uniform(Rmax_2d*geometric_factor)

        elif spatial == 'uniformnfw':
            Rmax_2d = model['spatial'][sub_mod_index][1][0]
            Rmax_z = model['spatial'][sub_mod_index][1][1]

            trunc_rad = None
            geometric_factor = self.cosmology.D_ratio([0, zplane], [0, self.zd])

            geometric_factor = 1
            self.spatial = Uniform(Rmax_2d * geometric_factor)


            self.spatial = Uniform_2d_1(Rmax_2d*geometric_factor, Rmax_z / self.cosmology.kpc_per_asec(zplane), rc=25)


        mass_func_type = model['massfunctype'][sub_mod_index]

        if mass_func_type == 'plaw':
            if model['norms'][sub_mod_index][0] == 'znorm':
                normalization = RedshiftNormalization(self.zd,self.zsrc)
                norm_kwargs = {}
                norm_kwargs['z'] = zplane
                norm_kwargs['dz'] = dz
                norm_kwargs['area'] = self.spatial.area
                norm_kwargs['mlow_norm'] = 10**model['args'][sub_mod_index][2]
                norm_kwargs['mhigh_norm'] = 10 ** model['args'][sub_mod_index][3]
            else:

                assert isinstance(model['norms'][sub_mod_index][0],float)
                norm_kwargs = {}
                normalization = model['norms'][sub_mod_index][0]

            subs = Plaw(norm=normalization,
                        logmL=model['mlow'][sub_mod_index],
                        logmH=model['mhigh'][sub_mod_index],
                        logmbreak=model['args'][sub_mod_index][0], scrit=self.cosmology.sigmacrit,
                        area=self.spatial.area,norm_kwargs=norm_kwargs)

            masses = subs.draw()

            Nhalos = int(np.shape(masses)[0])

            r3d, xpos, ypos = self.spatial.draw(N=Nhalos)

            self.xpos, self.ypos, self.r3d, self.Nhalos, self.masses = xpos, ypos, r3d, np.shape(masses)[
                    0], masses

            if trunc_rad is None:
                self.rtrunc = truncation(self.r3d, 1, self.masses, self.cosmology.sigmacrit)
            else:
                self.rtrunc = np.ones_like(self.masses)*trunc_rad

            assert len(self.rtrunc) == len(self.masses)
            assert len(self.rtrunc) == len(self.xpos)
            assert len(self.rtrunc) == self.Nhalos

            prof = model['proftypes'][sub_mod_index]

            for i in range(0, self.Nhalos):

                subhalo_args = {}

                if prof == 'TNFW':

                    if model['args'][sub_mod_index][1] == 0:
                        c_turnover = False
                    else:
                        c_turnover = True

                    lensmod = TNFW.TNFW(z1=self.zd, z2=self.zsrc, c_turnover=c_turnover, cosmology=self.cosmology)

                    subhalo_args['trunc'] = self.rtrunc[i]
                    subhalo_args['mhm'] = model['args'][sub_mod_index][0]

                elif prof == 'NFW':

                    if model['args'][sub_mod_index][1] == 0:
                        c_turnover = False
                    else:
                        c_turnover = True

                    lensmod = NFW.NFW(z1=self.zd, z2=self.zsrc, c_turnover=c_turnover, cosmology=self.cosmology)

                    subhalo_args['mhm'] = model['args'][sub_mod_index][0]

                elif prof == 'pjaffe':
                    raise StandardError('Pjaffe subhalos not yet implemented')

                    if model['args'][sub_mod_index][1] == 0:
                        core = 1e-6
                    else:
                        core = model['args'][sub_mod_index][1]

                    lensmod = PJaffe.PJaffe()

                elif prof == 'ptmass':

                    lensmod = PointMass.PointMass()

                else:

                    raise ValueError('profile '+prof+' not valid, supply valid mass profile for halos')

                subhalos.append(Deflector(use_lenstronomy_halos = self.use_lenstronomy_halos, subclass=lensmod, x=self.xpos[i],
                                  y=self.ypos[i], mass=self.masses[i], redshift=zplane, is_subhalo=True, **subhalo_args))

        else:
            raise StandardError('other mass function types not yet implemented')

        return subhalos

    def draw_subhalos(self,N_real=1,**kwargs):

        # returns a list of lists; each nested list is composed of a single substructure realization
        # If there are multiple lens planes, each nested list is composed of a list of realizations in different planes
        # e.g. [[realization1],[realization2],...[realizationN]]; realization1 = [plane1,plane2... planeK]
        # plane1 = [subhalo1,subhalo2,subhalo3...]
        #  N_real lists of physical parameters (position, masses, truncation, etc.)
        #  for subhalos, whose functional form is not yet specified

        # drawn according to self.mass_functions

        assert self.substrucutre_initialized

        Nmods = self.substructure_model['Nprofiles']

        substrucuture_realizations = []

        for n in range(0,N_real):

            subhalos_inplane = []

            for modnum in range(0,Nmods):

                if self.substructure_model['Nplanes'][modnum] == 1:
                    multiplane = False

                else:
                    multiplane = True

                if multiplane:

                    redshifts = np.linspace(self.substructure_model['zlow'][modnum],self.substructure_model['zhigh'][modnum],
                                            self.substructure_model['Nplanes'][modnum]+1)[:-1]

                    kwargs['dz'] = redshifts[1] - redshifts[0]

                    if redshifts[0]==0:
                        redshifts = redshifts[1:]

                    for zval in redshifts:

                        subhalos_inplane += self.add_single_plane(zval,model=self.substructure_model,sub_mod_index=modnum,**kwargs)

                else:
                    subhalos_inplane += self.add_single_plane(self.zd,model=self.substructure_model,sub_mod_index=modnum,**kwargs)

            substrucuture_realizations.append(subhalos_inplane)

        return substrucuture_realizations

    def set_substructure_model(self,models):
        """

        :param model: Defines a model for a substructure realization(s)
        :return: model kwargs

        model syntax: Nprofiles_massfunc1_prof1_norm1_mlow1_mhigh1_args1_spatial1_Nplanes_z1_z2+massfunc2_...

        args form: ['mbreak',extra]
        for NFW: extra is c turnover 0 is off 1 is on

        spatial1 form: ['name',args]
        if 'name'== 'uniformnfw' args = [Rmax_2d,Rmax_z]

        if 'name'== 'uniformflat' args = [Rmax_2d,trunc]

        Nprofiles specifies the number of different substructure models per realization

        Nplanes refers to multiple lens planes. If Nplanes>1, must specify:
        ..._Nplanes_zlow_zhigh_zstep

        """

        self.substrucutre_initialized = True
        realization_args = {}
        realization_args['massfunctype'] = []
        realization_args['proftypes'] = []
        realization_args['norms'] = []
        realization_args['mlow'] = []
        realization_args['mhigh'] = []
        realization_args['args'] = []
        realization_args['spatial'] = []
        realization_args['Nplanes'] = []
        realization_args['zlow'] = []
        realization_args['zhigh'] = []
        realization_args['zstep'] = []

        realization_args['Nprofiles'] = len(models)

        for model in models:
            count = 0

            splitname = model.split('_')

            # mlow_high should be in log(mass)
            realization_args['massfunctype'].append(splitname[count])
            realization_args['proftypes'].append(splitname[count+1])
            realization_args['norms'].append(eval(splitname[count+2]))
            realization_args['mlow'].append(float(splitname[count+3]))
            realization_args['mhigh'].append(float(splitname[count+4]))
            realization_args['args'].append(eval(splitname[count+5]))
            realization_args['spatial'].append(eval(splitname[count+6]))

            if int(splitname[count+7]) != 1:

                realization_args['Nplanes'].append(int(splitname[count+7]))
                realization_args['zlow'].append(float(splitname[count + 8]))
                realization_args['zhigh'].append(float(splitname[count + 9]))
                count+=9
            else:
                realization_args['Nplanes'].append(1)
                realization_args['zlow'].append(self.zd)
                realization_args['zhigh'].append(self.zsrc)
                realization_args['zstep'].append(1)
                count+=7

        return realization_args

    def substructure_init(self,model):

        self.substructure_model = self.set_substructure_model(model)
        self.substrucutre_initialized = True

if False:
    d = HaloGen()
    #model syntax: massfunc1_prof1_norm1_mlow1_mhigh1_args1_Nplanes_massfunc2_...
    # args syntax: [mbreak,core/c(m) relation,logmlow_norm,logmhigh_norm]
    models = ["plaw_tNFW_[.0005]_6_10_[0,1]_['uniformnfw',[3,500]]_1"]
    #models = ["plaw_tNFW_['znorm']_6_10_[0,1,2,10]_['uniformnfw',[3,500]]_20_0_0.5"]

    d.substructure_init(model=models)

    realization = d.draw_subhalos(N_real=1)
    print (realization[0][0].lensclass.params)





