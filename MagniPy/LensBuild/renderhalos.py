from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from spatial_distribution import *
from massfunctions import *
from MagniPy.MassModels import TNFW
from MagniPy.MassModels import NFW
from MagniPy.MassModels import PJaffe
from MagniPy.MassModels import PointMass
from MagniPy.LensBuild.lens_assemble import Deflector
from BuildRoutines.halo_environments import *

class HaloGen:

    def __init__(self,zd=None,zsrc=None):

        """

        :param zd: main deflector redshift
        :param zsrc: source redshift
        """

        self.cosmology = Cosmo(zd=zd,zsrc=zsrc)

        self.zd,self.zsrc = self.cosmology.zd,self.cosmology.zsrc

    def draw_model(self, model_kwargs=[], model_name='', spatial_name='', massprofile='', Nrealizations=1, rescale_sigma8=False):

        """
        Main execution routine for drawing (sub)halo realizations.

        returns: a list with 'Nrealizations' elements; each element is a halo realization specified by the other
                    input arguments discussed below.

        ###############################################################################################################

        0) model_kwargs: a dictionary containing keyword arguments related to the following mass function and spatial
                        distribution models

        ###############################################################################################################

        1) model_name:
        - Can be 'plaw_main', 'plaw_LOS', 'composite_plaw', 'delta_main', 'delta_LOS', 'composite_delta'

            A) plaw_main: a power law mass function at the main deflector reshift.

                model_kwargs:

                    Mandatory inputs:
                    i. fsub: substructure mass fraction (in projection) at the Einstein radius

                    Optional inputs:
                    i. kappa_Rein: convergence in substructure at the Einstein radius (1 arcsecond); defaults to 0.5
                    ii. plaw_index: power law index; defaults to -1.9
                    iii. turnover_index: mass function turnover power law index below mhm; defaults to 1.3
                    iv. logmhm: log10 of half mode mass scale; defaults to 0
                        (anything other than zero will yield m_hm = 10^logmhm)
                    v. log_mL: log10 of minimum subhalo mass; defaults to 6
                    vi. log_mH: log10 of minimum subhalo mass; defaults to 10

            B) plaw_LOS: a power law mass function between redshift zmin and zmax

                model_kwargs:

                    Mandatory inputs:
                    i. zmin: start redshift
                    ii. zmax: end resfhit

                    Optional inputs:
                    i. turnover_index: mass function turnover power law index below mhm; defaults to 1.3
                    ii. logmhm: log10 of half mode mass scale; defaults to 0
                        (anything other than zero will yield m_hm = 10^logmhm)
                    iii. rescale_sigma8 (bool): flag to rescale cosmology to account for an under dense region
                    iv. omega_M_void (float): matter density in an underdense region; only used if rescale_sigma8 is True
                    v. log_mL: log10 of minimum subhalo mass; defaults to 6
                    vi. log_mH: log10 of minimum subhalo mass; defaults to 10

            C) composite_plaw: A combination of 'plaw_LOS' and 'plaw_main'; see above documentation

            D) delta_LOS: delta function mass function between zmin and zmax

                model_kwargs:

                    Mandatory inputs:
                    i. M: mass on which to center delta function
                    ii. matter_fraction: fraction of the matter density 'omega_M' composed of masses 'M'
                    iii. zmin: minimum redshift
                    iv. zmax: maximum redshift

        ###############################################################################################################

        2) spatial_name:
        - Can be 'uniform2d', 'uniform_cored_nfw'

            A) uniform2d: uniform spatial distribution inside a circle

                model_kwargs:

                    Optional inputs:
                    i. rmax2d_asec: maximum circular radius in which to render halos; defaults to 3 arcseconds

            B) uniform_cored_nfw: an approximation to an NFW profile projected in two dimensions

                rho_2d ~ (1+rc^2/r^2)^-1
                rho_3d ~ (1+(z^2+rc^2) / r^2 )^(-3/2)

                model_kwargs:

                    Optional inputs:
                    i. rmax2d_asec: maximum circular radius in which to render halos; defaults to 3 arcseconds
                    ii. rmaxz_kpc: maximum z radius to render halos; basically the virial radius; defaults to 500kpc
                    iii. nfw_core_kpc: core size in kpc; defaults to 100 kpc

        ###############################################################################################################

        3) massprofiles:
        - Can be 'NFW', 'TNFW', 'PJaffe', 'PointMass'

            see related documentation in MagniPy.MassModels

        """

        ###############################################################################################################
        ############################################## MAIN PROGRAM ###################################################
        ###############################################################################################################

        def _return_delta_LOS(_spatial_):

            mass_function_type = []
            spatial_distribution_type = []

            N,zvals = LOS_delta(model_kwargs['M'],model_kwargs['matter_fraction'],zmin=model_kwargs['zmin'], zmax=model_kwargs['zmax'],
                                                 zmain=self.cosmology.zd, zsrc=self.cosmology.zsrc)

            modelkwargs['N'] = np.random.poisson(N)
            model_kwargs['logmass'] = np.log10(model_kwargs['M'])

            mass_function_type.append('delta')
            spatial_distribution_type.append(_spatial_)

            halos = self._halos(mass_function_type=mass_function_type, spatial_distribution=spatial_distribution_type,
                                redshift=self.cosmology.zd, Nrealizations=Nrealizations, mass_profile=[massprofile],
                                modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs])
            return halos

        def _return_plaw_main(_spatial_):

            mass_function_type = []
            spatial_distribution_type = []

            if 'kappa_Rein' in model_kwargs:
                kappaRein = model_kwargs['kappa_Rein']
            else:
                kappaRein = kappa_Rein

            if 'plaw_index' in model_kwargs:
                modelkwargs['plaw_index'] = model_kwargs['rmax2d_asec']
            else:
                modelkwargs['plaw_index'] = powerlaw_defaults['plaw_index']

            if 'turnover_index' in model_kwargs:
                modelkwargs['turnover_index'] = model_kwargs['turnover_index']
            else:
                modelkwargs['turnover_index'] = powerlaw_defaults['turnover_index']

            if 'logmhm' in model_kwargs:
                modelkwargs['logmhm'] = model_kwargs['logmhm']
            else:
                modelkwargs['logmhm'] = 0

            if 'log_mL' in model_kwargs:
                modelkwargs['log_mL'] = model_kwargs['log_mL']
            else:
                modelkwargs['log_mL'] = powerlaw_defaults['log_ML']

            if 'log_mH' in model_kwargs:
                modelkwargs['log_mH'] = model_kwargs['log_mH']
            else:
                modelkwargs['log_mH'] = powerlaw_defaults['log_MH']

            A0, _ = mainlens_plaw(model_kwargs['fsub'], plaw_index=modelkwargs['plaw_index'], cosmo=self.cosmology,
                                  kappa_Rein=kappaRein, log_mL = modelkwargs['log_mL'], log_mH = modelkwargs['log_mH'])

            modelkwargs['normalization'] = A0

            mass_function_type.append('plaw')
            spatial_distribution_type.append(_spatial_)

            halos = self._halos(mass_function_type=mass_function_type, spatial_distribution=spatial_distribution_type,
                                redshift=self.cosmology.zd, Nrealizations=Nrealizations, mass_profile=[massprofile],
                                modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs])
            return halos

        def _return_plaw_LOS(_spatial_):

            mass_function_type = []
            spatial_distribution_type = []
            halos = []

            if 'turnover_index' in model_kwargs:
                modelkwargs['turnover_index'] = model_kwargs['turnover_index']
            else:
                modelkwargs['turnover_index'] = powerlaw_defaults['turnover_index']

            if 'logmhm' in model_kwargs:
                modelkwargs['logmhm'] = model_kwargs['logmhm']
            else:
                modelkwargs['logmhm'] = 0

            if 'rescale_sigma8' in model_kwargs:
                rescale_sigma8 = True
                omega_M_void = model_kwargs['omega_M_void']
            else:
                rescale_sigma8 = False
                omega_M_void = None

            if 'log_mL' in model_kwargs:
                modelkwargs['log_mL'] = model_kwargs['log_mL']
            else:
                modelkwargs['log_mL'] = powerlaw_defaults['log_ML']

            if 'log_mH' in model_kwargs:
                modelkwargs['log_mH'] = model_kwargs['log_mH']
            else:
                modelkwargs['log_mH'] = powerlaw_defaults['log_MH']

            A0_z, plaw_index_z, zvals = LOS_plaw(zmin=model_kwargs['zmin'], zmax=model_kwargs['zmax'],
                                                 zmain=self.cosmology.zd, zsrc=self.cosmology.zsrc,
                                                 rescale_sigma8 = rescale_sigma8, omega_M_void=omega_M_void,
                                                 log_mL=modelkwargs['log_mL'],log_mH=modelkwargs['log_mH'])

            mass_function_type.append('plaw')
            spatial_distribution_type.append(_spatial_)

            for N in range(0, Nrealizations):

                _plane_halos = []

                for p in range(0, len(A0_z)):
                    modelkwargs['normalization'] = A0_z[p]
                    model_kwargs['plaw_index'] = plaw_index_z
                    redshift = zvals[p]

                    _plane_halos += self._halos(mass_function_type=mass_function_type,
                                                spatial_distribution=spatial_distribution_type, redshift=redshift,
                                                Nrealizations=1, mass_profile=[massprofile],
                                                modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs])[0]

                halos.append(_plane_halos)

            return halos


        if model_name not in ['plaw_main','plaw_LOS','composite_plaw','delta_main','delta_LOS','composite_delta']:
                raise Exception('model name not recognized')
        if spatial_name not in ['uniform2d','uniform_cored_nfw']:
                raise Exception('spatial distribution not recognized')

        if isinstance(massprofile, list):
            raise Exception('not yet implemented')
        else:
            if massprofile not in ['NFW', 'TNFW', 'PTmass', 'PJaffe']:
                    raise Exception('mass profile not recognized')


        modelkwargs = {}
        spatialkwargs = {}

        if 'rmax2d_asec' in model_kwargs:
            spatialkwargs['rmax2d'] = model_kwargs['rmax2d_asec']
        else:
            spatialkwargs['rmax2d'] = spatial_defaults['theta_max']

        if spatial_name == 'uniform_cored_nfw':
            if 'rmaxz_kpc' in model_kwargs:
                spatialkwargs['rmaxz'] = model_kwargs['Rmax_z_kpc'] * self.cosmology.kpc_per_asec(
                    self.cosmology.zd) ** -1
            else:
                spatialkwargs['rmaxz'] = spatial_defaults['Rmax_z_kpc'] * self.cosmology.kpc_per_asec(
                    self.cosmology.zd) ** -1

            if 'nfw_core_kpc' in model_kwargs:
                spatialkwargs['rc'] = model_kwargs['nfw_core_kpc'] * self.cosmology.kpc_per_asec(
                    self.cosmology.zd) ** -1
            else:
                spatialkwargs['rc'] = spatial_defaults['nfw_core_kpc'] * self.cosmology.kpc_per_asec(
                    self.cosmology.zd) ** -1

        if model_name == 'plaw_main':

            HALOS = _return_plaw_main( _spatial_ = spatial_name)

        elif model_name == 'plaw_LOS':

            HALOS = _return_plaw_LOS(_spatial_ = 'uniform2d')

        elif model_name=='composite_plaw':

            HALOS_main = _return_plaw_main(_spatial_ = spatial_name)
            HALOS_LOS = _return_plaw_LOS(_spatial_ = 'uniform2d')

            HALOS = []

            for n in range(0,Nrealizations):

                HALOS.append(HALOS_LOS[n]+HALOS_main[n])

        elif model_name == 'delta_main':
            raise Exception('not yet implemented')

        elif model_name == 'delta_LOS':
            HALOS = _return_delta_LOS()

        return HALOS

    def _halos(self,mass_function_type=None,spatial_distribution=None,redshift=None,Nrealizations=1,
                   mass_profile=None,modelkwargs={},spatialkwargs={}):
        """
        :param mass_func_type: "plaw, delta, etc."
        :param spatial: 'uniformflat','uniformnfw'
        :param redshift: redshift of the plane
        :param modelkwargs: keyword args for a particular model
        :return:
        """

        assert isinstance(mass_function_type,list)
        assert isinstance(spatial_distribution,list)
        assert len(mass_function_type)==len(spatial_distribution)

        realizations = []

        for r in range(0,Nrealizations):

            subhalos = []

            for i in range(0,len(mass_function_type)):

                mass_func_type = mass_function_type[i]
                spatial_type = spatial_distribution[i]
                massprofile= mass_profile[i]

                if mass_func_type == 'plaw':

                    massfunction = Plaw(**modelkwargs[i])

                elif mass_func_type == 'delta':

                    massfunction = Delta(**modelkwargs[i])

                else:
                    if mass_func_type is None:
                        raise Exception('supply mass function type')
                    else:
                        raise Exception('mass function type '+str(mass_func_type)+' not recognized')

                if spatial_type == 'uniform2d':

                    if 'rmaxz' in spatialkwargs[i]:
                        del spatialkwargs[i]['rmaxz']
                    if 'rc' in spatialkwargs[i]:
                        del spatialkwargs[i]['rc']

                    spatial= Uniform_2d(cosmology=self.cosmology,**spatialkwargs[i])

                elif spatial_type == 'uniform_cored_nfw':

                    spatial = Uniform_cored_nfw(cosmology=self.cosmology,**spatialkwargs[i])

                else:
                    if spatial_type is None:
                        raise Exception('supply spatial distribution type')
                    else:
                        raise Exception('spatial distribution ' + str(mass_func_type) + ' not recognized')

                masses = massfunction.draw()

                R,x,y = spatial.draw(int(len(masses)),redshift)

                for j in range(0, int(len(masses))):

                    subhalo_args = {}

                    if massprofile == 'TNFW':

                        c_turnover = concentration_turnover

                        lensmod = TNFW.TNFW(z=redshift, zsrc=self.cosmology.zsrc, c_turnover=c_turnover)

                        if spatial_distribution[i] == 'uniform2d':
                            truncation = TruncationFuncitons(truncation_routine='NFW_m200')
                            subhalo_args['trunc'] = None

                        elif spatial_distribution[i] == 'uniform_cored_nfw':
                            truncation = TruncationFuncitons(truncation_routine='tidal_3d')
                            subhalo_args['trunc'] = truncation.function(mass=masses,r3d=R,
                                                                        sigmacrit=self.cosmology.sigmacrit)

                        subhalo_args['mhm'] = modelkwargs[i]['logmhm']

                    elif massprofile == 'NFW':

                        c_turnover = concentration_turnover

                        lensmod = NFW.NFW(z=redshift, zsrc = self.cosmology.zsrc, c_turnover=c_turnover)

                        subhalo_args['mhm'] = modelkwargs[i]['logmhm']

                    elif massprofile == 'pjaffe':

                        if spatial_distribution[i] == 'uniform2d':
                            truncation = TruncationFuncitons(truncation_routine='gaussian')
                            subhalo_args['trunc'] = truncation.function(mean=0.1,sigma=0.05,size=len(masses))

                        elif spatial_distribution[i] == 'uniform_cored_nfw':
                            truncation = TruncationFuncitons(truncation_routine='tidal_3d')
                            subhalo_args['trunc'] = truncation.function(mass=masses, r3d=R,
                                                                        sigmacrit=self.cosmology.sigmacrit)

                        lensmod = PJaffe.PJaffe(z=redshift,zsrc = self.cosmology.zsrc)

                    elif massprofile == 'ptmass':

                        lensmod = PointMass.PointMass(z=redshift,zsrc = self.cosmology.zsrc)

                    else:

                        raise ValueError('profile '+massprofile+' not valid, supply valid mass profile for halos')

                    subhalos.append(Deflector(subclass=lensmod, x=x[j],
                                  y=y[j], mass=masses[j], redshift=redshift, is_subhalo=True, **subhalo_args))


            realizations.append(subhalos)

        return realizations

    def get_masses(self,realization_list):

        realization_masses = []

        for realization in realization_list:

            realization_masses.append(np.array([deflector.other_args['mass'] for deflector in realization]))

        return realization_masses


if False:
    render = HaloGen(zd=.5,zsrc=1.5)

    model_args = {}
    model_args['fsub'] = 0.1
    model_args['zmin'] = 0.001
    model_args['zmax'] = .57

    halos = render.draw_model(model_name='plaw_main', spatial_name='uniform_cored_nfw',
                              massprofile='NFW', model_kwargs=model_args, Nrealizations=1)[0]
    m = []

    for halo in halos:
        m.append(np.log10(halo.other_args['mass']))
    h,b = np.histogram(m)
    print np.polyfit((b[0:-1]),np.log10(h),1)






