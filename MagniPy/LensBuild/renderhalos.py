from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from spatial_distribution import *
from massfunctions import *
from MagniPy.MassModels import TNFW
from MagniPy.MassModels import NFW
from MagniPy.MassModels import PJaffe
from MagniPy.MassModels import PointMass
from MagniPy.MassModels import uniformsheet
from MagniPy.LensBuild.lens_assemble import Deflector
from BuildRoutines.halo_environments import *
from MagniPy.util import filter_by_position
from MagniPy.LensBuild.BuildRoutines import PBHgen
from copy import deepcopy
from halo_truncations import Truncation

class HaloGen:

    def __init__(self,zd=None,zsrc=None):

        """

        :param zd: main deflector redshift
        :param zsrc: source redshift
        """

        self.cosmology = Cosmo(zd=zd,zsrc=zsrc)

        self.zd,self.zsrc = self.cosmology.zd,self.cosmology.zsrc

        self.A0_z = None
        self.plaw_index_z = None
        self.redshift_values = None

    def draw_model(self, model_kwargs=[], model_name='', spatial_name='', massprofile='', Nrealizations=1,
                   rescale_sigma8=False, filter_halo_positions=False, **filter_kwargs):

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

        filter_halo_positions (bool): flag to filter subhalos by position
        filterkwargs: 'x_filter','y_filter',

        """

        ############################################## MAIN PROGRAM ###################################################

        if model_name not in ['plaw_main','plaw_LOS','composite_plaw','delta_LOS']:
                raise Exception('model name not recognized')
        if spatial_name not in ['uniform2d','uniform_cored_nfw']:
                raise Exception('spatial distribution not recognized')

        if isinstance(massprofile, list):
            raise Exception('not yet implemented')
        else:
            if massprofile not in ['NFW', 'TNFW', 'PTmass', 'PJaffe']:
                    raise Exception('mass profile not recognized')


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

        position_filter_kwargs = {}

        if filter_halo_positions:

            assert 'x_filter' in filter_kwargs
            assert 'y_filter' in filter_kwargs

            position_filter_kwargs['x_position'] = filter_kwargs['x_filter']
            position_filter_kwargs['y_position'] = filter_kwargs['y_filter']

            position_filter_kwargs['filter_halo_positions'] = True

            if 'mindis' in filter_kwargs:
                position_filter_kwargs['mindis'] = filter_kwargs['mindis']
            else:
                position_filter_kwargs['mindis'] = filter_args['mindis']

            if 'log_masscut_low' in filter_kwargs:
                position_filter_kwargs['log_masscut_low'] = filter_kwargs['log_masscut_low']
            else:
                position_filter_kwargs['log_masscut_low'] = filter_args['log_masscut_low']

        if model_name == 'plaw_main':

            HALOS = self._return_plaw_main(_spatial_ = spatial_name,position_filter_kwargs=position_filter_kwargs,
                                           model_kwargs=model_kwargs,massprofile=massprofile,spatialkwargs=spatialkwargs,
                                           Nrealizations=Nrealizations)

        elif model_name == 'plaw_LOS':

            cone_base = spatial_defaults['default_cone_base_factor']*spatialkwargs['rmax2d']

            HALOS = self._return_plaw_LOS(_spatial_ = 'uniform2d',position_filter_kwargs=position_filter_kwargs,
                                           model_kwargs=model_kwargs,massprofile=massprofile,spatialkwargs=spatialkwargs,
                                          cone_base=cone_base,Nrealizations=Nrealizations)

        elif model_name=='composite_plaw':

            cone_base = spatial_defaults['default_cone_base_factor'] * spatialkwargs['rmax2d']

            HALOS_main = self._return_plaw_main(_spatial_ = spatial_name,position_filter_kwargs=position_filter_kwargs,
                                           model_kwargs=model_kwargs,massprofile=massprofile,spatialkwargs=spatialkwargs,
                                           Nrealizations=Nrealizations)

            HALOS_LOS = self._return_plaw_LOS(_spatial_ = 'uniform2d',position_filter_kwargs=position_filter_kwargs,
                                           model_kwargs=model_kwargs,massprofile=massprofile,spatialkwargs=spatialkwargs,
                                              cone_base=cone_base, Nrealizations=Nrealizations)

            HALOS = []

            for n in range(0,Nrealizations):

                HALOS.append(HALOS_LOS[n]+HALOS_main[n])

        elif model_name == 'delta_main':
            raise Exception('not yet implemented')

        elif model_name == 'delta_LOS':
            raise Exception('not yet implemented')

        elif model_name == 'composite_delta':
            raise Exception('not yet implemented')

        return HALOS

    def _return_delta_LOS(self,_spatial_,position_filter_kwargs,model_kwargs,massprofile,spatialkwargs,cone_base,Nrealizations):

        mass_function_type = []
        spatial_distribution_type = []
        modelkwargs = {}

        Nz, zvals = LOS_delta(model_kwargs['M'], model_kwargs['matter_fraction'], zmin=model_kwargs['zmin'],
                             zmax=model_kwargs['zmax'],
                             zmain=self.cosmology.zd, zsrc=self.cosmology.zsrc,cone_base=cone_base)

        model_kwargs['logmass'] = np.log10(model_kwargs['M'])

        mass_function_type.append('delta')
        spatial_distribution_type.append(_spatial_)

        for N in range(0, Nrealizations):

            _plane_halos = []

            for p in range(0, len(zvals)):

                modelkwargs['N'] = np.random.poisson(Nz[p])

                _plane_halos += self._halos(cosmo_at_plane=Cosmo(zd=zvals[p],zsrc=self.zsrc,compute=False),mass_function_type=mass_function_type, spatial_distribution=spatial_distribution_type,
                            redshift=zvals[p], Nrealizations=Nrealizations, mass_profile=[massprofile],
                            modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs], **position_filter_kwargs)[0]

            _plane_halos.append(_plane_halos)

        return _plane_halos

    def _return_plaw_main(self,_spatial_,position_filter_kwargs,model_kwargs,massprofile,spatialkwargs,Nrealizations):

        mass_function_type = []
        spatial_distribution_type = []
        modelkwargs = deepcopy(model_kwargs)

        if 'kappa_Rein' not in modelkwargs:
            modelkwargs['kappa_Rein'] = kappa_Rein_default

        if 'plaw_index' not in modelkwargs:
            modelkwargs['plaw_index'] = powerlaw_defaults['plaw_index']

        if 'turnover_index' not in modelkwargs:
            modelkwargs['turnover_index'] = powerlaw_defaults['turnover_index']

        if 'logmhm' not in modelkwargs:
            modelkwargs['logmhm'] = 0

        if 'log_mL' not in modelkwargs:
            modelkwargs['log_mL'] = powerlaw_defaults['log_ML']

        if 'log_mH' not in modelkwargs:
            modelkwargs['log_mH'] = powerlaw_defaults['log_MH']

        if 'fsub' in modelkwargs:

            A0_perasec, _ = mainlens_plaw(fsub=modelkwargs['fsub'],plaw_index=modelkwargs['plaw_index'],cosmo=self.cosmology,
                                  kappa_Rein=modelkwargs['kappa_Rein'], log_mL = modelkwargs['log_mL'], log_mH = modelkwargs['log_mH'])
            A0 = A0_perasec*np.pi*spatialkwargs['rmax2d']**2

        elif 'A0_perasec' in modelkwargs:

            A0 = modelkwargs['A0_perasec']*np.pi*spatialkwargs['rmax2d']**2

        else:
            raise Exception('either fsub or A0_perasec must be specified for plaw_main')

        modelkwargs['normalization'] = A0

        mass_function_type.append('plaw')
        spatial_distribution_type.append(_spatial_)

        halos = self._halos(cosmo_at_plane=self.cosmology,mass_function_type=mass_function_type, spatial_distribution=spatial_distribution_type,
                            redshift=self.cosmology.zd, Nrealizations=Nrealizations, mass_profile=[massprofile],
                            modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs], **position_filter_kwargs)

        return halos

    def _return_plaw_LOS(self,_spatial_,position_filter_kwargs,model_kwargs,massprofile,spatialkwargs,cone_base,Nrealizations):

        mass_function_type = []
        spatial_distribution_type = []
        halos = []
        modelkwargs = deepcopy(model_kwargs)

        if 'kappa_Rein' not in modelkwargs:
            modelkwargs['kappa_Rein'] = kappa_Rein_default

        if 'plaw_index' not in modelkwargs:
            modelkwargs['plaw_index'] = powerlaw_defaults['plaw_index']

        if 'turnover_index' not in modelkwargs:
            modelkwargs['turnover_index'] = powerlaw_defaults['turnover_index']

        if 'logmhm' not in modelkwargs:
            modelkwargs['logmhm'] = 0

        if 'log_mL' not in modelkwargs:
            modelkwargs['log_mL'] = powerlaw_defaults['log_ML']

        if 'log_mH' not in modelkwargs:
            modelkwargs['log_mH'] = powerlaw_defaults['log_MH']

        if 'rescale_sigma8' not in modelkwargs:
            modelkwargs['rescale_sigma8'] = False
            modelkwargs['omega_M_void'] = None

        if self.A0_z is None:
            A0_z, plaw_index_z, zvals = LOS_plaw(zmain=self.cosmology.zd, zsrc=self.cosmology.zsrc,zmin=modelkwargs['zmin'],
                                             zmax=modelkwargs['zmax'],rescale_sigma8=modelkwargs['rescale_sigma8'],
                                             omega_M_void=modelkwargs['omega_M_void'],log_mL=modelkwargs['log_mL'],
                                             log_mH=modelkwargs['log_mH'],cone_base=cone_base)
            self.A0_z = A0_z
            self.plaw_index_z = plaw_index_z
            self.redshift_values = zvals

        else:

            A0_z = self.A0_z
            plaw_index_z = self.plaw_index_z
            zvals = self.redshift_values

        mass_function_type.append('plaw')
        spatial_distribution_type.append(_spatial_)

        for N in range(0, Nrealizations):

            _plane_halos = []

            for p in range(0, len(A0_z)):
                modelkwargs['normalization'] = A0_z[p]
                modelkwargs['plaw_index'] = plaw_index_z[p]

                redshift = zvals[p]

                _plane_halos += self._halos(cosmo_at_plane=Cosmo(zd=zvals[p],zsrc=self.zsrc,compute=False),mass_function_type=mass_function_type,
                                            spatial_distribution=spatial_distribution_type, redshift=redshift,
                                            Nrealizations=1, mass_profile=[massprofile],
                                            modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs],add_mass_sheet=True,
                                            **position_filter_kwargs)[0]

            halos.append(_plane_halos)

        return halos

    def _return_delta_main(self, _spatial_, position_filter_kwargs, model_kwargs, massprofile, spatialkwargs, redshift,
                           Nrealizations):

        assert position_filter_kwargs['filter_halo_positions'] is True

        modelkwargs = {}

        mass_function_type = []
        spatial_distribution_type = []

        N,rmax2d = number_per_image(f_pbh=model_kwargs['f_PBH'], redshift=redshift, zsrc=self.cosmology.zsrc,
                             cosmology_class=self.cosmology, M=model_kwargs['M'], R_ein=None)

        spatialkwargs['rmax2d'] = rmax2d

        modelkwargs['N'] = np.random.poisson(N)

        modelkwargs['logmass'] = np.log10(model_kwargs['M'])

        mass_function_type.append('delta')
        spatial_distribution_type.append(_spatial_)

        halos = self._halos(cosmo_at_plane=self.cosmology,mass_function_type=mass_function_type, spatial_distribution=spatial_distribution_type,
                            redshift=self.cosmology.zd, Nrealizations=Nrealizations, mass_profile=[massprofile],
                            modelkwargs=[modelkwargs], spatialkwargs=[spatialkwargs])

        return halos

    def _halos(self,cosmo_at_plane=None,mass_function_type=None,spatial_distribution=None,redshift=None,Nrealizations=1,
                   mass_profile=None,modelkwargs={},spatialkwargs={},filter_halo_positions=False,add_mass_sheet=True,**kwargs):
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

                    massfunction = Plaw(normalization=modelkwargs[i]['normalization'], log_mL=modelkwargs[i]['log_mL'],
                                                log_mH=modelkwargs[i]['log_mH'], logmhm=modelkwargs[i]['logmhm'],
                                                plaw_index=modelkwargs[i]['plaw_index'], turnover_index=modelkwargs[i]['turnover_index'])

                    masses = massfunction.draw()

                    x, y, R, area = self.get_spatial(N=int(len(masses)), redshift=redshift, spatial_type=spatial_type,
                                               cosmo_at_plane=cosmo_at_plane, spatialkwargs=spatialkwargs[i])

                elif mass_func_type == 'delta':

                    massfunction = Delta(N = modelkwargs[i]['N'], logmass=modelkwargs[i]['logmass'])
                    masses = massfunction.draw()

                    x, y, R, area = self.get_spatial(N=int(len(masses)), redshift=redshift, spatial_type=spatial_type,
                                               cosmo_at_plane=cosmo_at_plane, spatialkwargs=spatialkwargs[i])

                elif mass_function_type == 'plaw_order2':

                    massfunction_primary = Plaw(normalization=modelkwargs[i]['normalization'], log_mL=modelkwargs[i]['log_mL'],
                                                log_mH=modelkwargs[i]['log_mH'], logmhm=modelkwargs[i]['logmhm'],
                                                plaw_index=modelkwargs[i]['plaw_index'], turnover_index=modelkwargs[i]['turnover_index'])

                    masses = massfunction_primary.draw()

                    x,y,R,area = self.get_spatial(N=int(len(masses)), redshift=redshift, spatial_type=spatial_type,
                                             cosmo_at_plane=cosmo_at_plane, spatialkwargs=spatialkwargs[i])

                    massfunction = Plaw_secondary(M_parent=masses,parent_r2d=R,x_locations=x,y_locations=y,log_mL=6.5,logmhm=modelkwargs[i]['logmhm'])

                    masses,x,y,R = massfunction.draw()

                else:
                    if mass_func_type is None:
                        raise Exception('supply mass function type')
                    else:
                        raise Exception('mass function type '+str(mass_func_type)+' not recognized')

                for j in range(0, int(len(masses))):

                    subhalo_args = {}

                    if massprofile == 'TNFW':

                        c_turnover = concentration_turnover

                        lensmod = TNFW.TNFW(cosmology=cosmo_at_plane, c_turnover=c_turnover)

                        if spatial_distribution[i] == 'uniform2d':

                            truncation = Truncation(truncation_routine='fixed_radius')

                        elif spatial_distribution[i] == 'uniform_cored_nfw':

                            truncation = Truncation(truncation_routine='virial3d',
                                                    params={'sigmacrit': cosmo_at_plane.get_sigmacrit(), 'Rein': 1,
                                                            'r3d': R[j]})

                        subhalo_args['truncation'] = truncation

                        subhalo_args['mhm'] = modelkwargs[i]['logmhm']

                    elif massprofile == 'NFW':

                        c_turnover = concentration_turnover

                        lensmod = NFW.NFW(cosmology=cosmo_at_plane, c_turnover=c_turnover)

                        subhalo_args['mhm'] = modelkwargs[i]['logmhm']

                    elif massprofile == 'pjaffe':

                        truncation = Truncation(truncation_routine='virial3d',
                                                params={'sigmacrit': cosmo_at_plane.get_sigmacrit(), 'Rein': 1,
                                                        'r3d': R})

                        subhalo_args['truncation'] = truncation

                        lensmod = PJaffe.PJaffe(cosmology=cosmo_at_plane)

                    elif massprofile == 'ptmass':

                        lensmod = PointMass.PointMass(cosmology=cosmo_at_plane)

                    else:

                        raise ValueError('profile '+massprofile+' not valid, supply valid mass profile for halos')

                    subhalos.append(Deflector(subclass=lensmod, x=x[j],
                                  y=y[j], mass=masses[j], redshift=redshift, is_subhalo=True, **subhalo_args))

            if filter_halo_positions:

                subhalos, _ = filter_by_position(subhalos,x_filter=kwargs['x_position'],y_filter=kwargs['y_position'],mindis=kwargs['mindis'],
                                                 log_masscut_low=kwargs['log_masscut_low'],zmain=self.cosmology.zd,cosmology=cosmo_at_plane)

            if add_mass_sheet and len(subhalos)>0:

                masses = [obj.other_args['mass'] for obj in subhalos]

                mass_in_plane = np.sum(masses)

                plane_kappa = (mass_in_plane*area**-1)*cosmo_at_plane.get_sigmacrit()**-1

                subhalos.append(Deflector(subclass=uniformsheet.MassSheet(),redshift=redshift,is_subhalo=True,kappa_ext=-1*plane_kappa))

            realizations.append(subhalos)

        return realizations

    def get_spatial(self,N=int,redshift=None,spatial_type='',cosmo_at_plane=None,spatialkwargs={}):

        if spatial_type == 'uniform2d':

            spatial = Uniform_2d(cosmology=cosmo_at_plane, rmax2d=spatialkwargs['rmax2d'])
            x, y, R = spatial.draw(N, redshift)

        elif spatial_type == 'uniform_cored_nfw':

            spatial = Uniform_cored_nfw(cosmology=cosmo_at_plane, **spatialkwargs)
            x, y, R = spatial.draw(N, redshift)

        elif spatial_type == 'localized_uniform':

            spatial = Localized_uniform(cosmology=cosmo_at_plane, **spatialkwargs)
            x, y, R = spatial.draw(N, redshift)

        else:
            if spatial_type is None:
                raise Exception('supply spatial distribution type')
            else:
                raise Exception('spatial distribution ' + str(spatial_type) + ' not recognized')

        area = np.pi*spatial.rmax2d**2

        return x,y,R,area

    def get_masses(self, realization_list, mass_range = None, specific_redshift = None):

        realization_masses = []

        if specific_redshift is None:
            for realization in realization_list:


                if mass_range is not None:
                    object_generator = (deflector.other_args['mass'] for deflector in realization
                                        if np.min(mass_range) <= deflector.other_args['mass'] < np.max(mass_range))

                    realization_masses.append(list(object_generator))

                else:
                    realization_masses.append(np.array([deflector.other_args['mass'] for deflector in realization]))

        else:
            for realization in realization_list:

                if mass_range is not None:
                    object_generator = (deflector.other_args['mass'] for deflector in realization
                                        if (np.min(mass_range) <= deflector.other_args['mass'] <
                                            np.max(mass_range) and
                                            deflector.redshift == specific_redshift))

                    realization_masses.append(list(object_generator))

                else:
                    object_generator = (deflector.other_args['mass'] for deflector in realization
                                        if (deflector.redshift == specific_redshift))

                    realization_masses.append(list(object_generator))

        return realization_masses







