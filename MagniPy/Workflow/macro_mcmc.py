import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE
from copy import deepcopy
import corner.corner
import time

class MacroMCMC(object):

    from_class = False

    def __init__(self, lensdata_class, optroutine):

        self.data = lensdata_class
        self.optroutine = optroutine

    def run(self, macromodel=None,
            tovary={'source_size_kpc': [0.01, 0.05], 'gamma': [1.95, 2.2]}, N=500,
            zlens=0.5, optimizer_kwargs={}, write_to_file=False, fname=None,
            satellite_kwargs = None, satellite_mass_model = None, flux_ratio_index = None):

        assert flux_ratio_index is not None
        flux_ratio_index = int(flux_ratio_index)
        inds = []
        for i in range(0,4):
            if i != flux_ratio_index:
                inds.append(i)
        inds = np.array(inds)

        if write_to_file is True:
            assert fname is not None

        rein, ellip, PA, shear, shearPA, gamma, source_size, cen_x, cen_y = \
            [], [], [], [], [], [], [], [], []

        satellite_x, satellite_y, satellite_thetaE, z_sat = [], [], [], []

        fluxes = None

        if macromodel is None:
            macromodel = get_default_SIE(zlens)

        for i in range(0, N):

            if i % 100 == 0:
                print(str(i) + ' of ' + str(N) + '... ')

            args = {}
            for argname in optimizer_kwargs.keys():
                args.update({argname: optimizer_kwargs[argname]})

            macro = deepcopy(macromodel)
            if satellite_mass_model is not None:
                satellite_redshift = [zlens]*len(satellite_mass_model)

            for pname in tovary.keys():

                if pname in ['satellite_theta_E', 'satellite_x', 'satellite_y', 'satellite_redshift']:
                    assert satellite_kwargs is not None
                    assert satellite_mass_model is not None

                    if pname == 'satellite_redshift':
                        satellite_redshift = [np.random.uniform(tovary[pname][0], tovary[pname][1])]*len(satellite_mass_model)
                    elif pname == 'satellite_theta_E':
                        satellite_kwargs.update({'theta_E': np.random.uniform(tovary[pname][0], tovary[pname][1])})
                    elif pname == 'satellite_x':
                        satellite_kwargs.update({'center_x': np.random.uniform(tovary[pname][0], tovary[pname][1])})
                    elif pname == 'satellite_y':
                        satellite_kwargs.update({'center_y': np.random.uniform(tovary[pname][0], tovary[pname][1])})

                else:
                    low, high = tovary[pname][0], tovary[pname][1]
                    macro.lenstronomy_args[pname] = np.random.uniform(low, high)

            if 'gamma' in tovary.keys():
                gammavalue = np.random.uniform(tovary['gamma'][0], tovary['gamma'][1])
                macro.lenstronomy_args['gamma'] = gammavalue
            if 'source_size_kpc' in tovary.keys():
                srcsize = np.random.uniform(tovary['source_size_kpc'][0],
                                            tovary['source_size_kpc'][1])

            else:
                assert 'source_size_kpc' in optimizer_kwargs.keys()
                srcsize = optimizer_kwargs['source_size_kpc']
            args.update({'source_size_kpc': srcsize})


            if satellite_mass_model is not None:
                satellites = {'lens_model_name': satellite_mass_model, 'z_satellite': satellite_redshift,
                              'kwargs_satellite': [satellite_kwargs]}
            else:
                satellites = None

            args.update({'macromodel': macro})
            args.update({'satellites': satellites})

            if self.from_class:
                optdata, optmodel = self._optimize_once_fromlensclass(self.optroutine, args)

            else:
                out = self._optimize_once(self.optroutine, args)
                optdata, optmodel = out[0][0], out[1][0]

            if 'satellite_theta_E' in tovary.keys():
                satellite_thetaE.append(optmodel.satellite_kwargs[0]['theta_E'])
            if 'satellite_x' in tovary.keys():
                satellite_x.append(optmodel.satellite_kwargs[0]['center_x'])
            if 'satellite_y' in tovary.keys():
                satellite_y.append(optmodel.satellite_kwargs[0]['center_y'])
            if 'satellite_redshift' in tovary.keys():
                z_sat.append(optmodel.satellite_redshift[0])

            optimized_macromodel = optmodel.lens_components[0]
            modelargs = optimized_macromodel.lenstronomy_args

            rein.append(modelargs['theta_E'])

            ellipPA = optimized_macromodel.ellip_PA_polar()
            ellip.append(ellipPA[0])
            PA.append(ellipPA[1])

            shear.append(optimized_macromodel.shear)
            shearPA.append(optimized_macromodel.shear_theta)

            gamma.append(gammavalue)
            source_size.append(srcsize * 1000)
            cen_x.append(modelargs['center_x'])
            cen_y.append(modelargs['center_y'])

            if fluxes is None:
                norm = optdata.m[flux_ratio_index]
                fluxes = optdata.m[inds] * norm**-1
            else:
                norm = optdata.m[flux_ratio_index]
                new_ratios = optdata.m[inds] * norm**-1
                fluxes = np.vstack((fluxes, new_ratios))

        rein = np.round(rein, 4)
        ellip = np.round(ellip, 4)
        PA = np.round(PA, 2)
        shear = np.round(shear, 4)
        shearPA = np.round(shearPA, 3)
        gamma = np.round(gamma, 2)
        source_size = np.round(source_size, 2)
        cen_x, cen_y = np.round(cen_x, 4), np.round(cen_y, 4)
        z_sat = np.round(z_sat, 2)
        satellite_x, satellite_y = np.round(satellite_x, 4), np.round(satellite_y, 4)
        satellite_thetaE = np.round(satellite_thetaE, 3)

        full_dictionary = {'Rein': rein, 'ellip': ellip, 'ellipPA': PA,
                           'shear': shear, 'shearPA': shearPA, 'gamma': gamma,
                           'srcsize': source_size, 'centroid_x': cen_x, 'centroid_y': cen_y,
                           'G2x': satellite_x, 'G2y': satellite_y, 'G2thetaE': satellite_thetaE,
                           'G2redshift': z_sat}

        if write_to_file:
            array = fluxes
            #header = 'f1 f2 f3 '
            for key in full_dictionary.keys():
                if len(full_dictionary[key]) == 0:
                    continue
            #    header += key + ' '
                array = np.column_stack((array, full_dictionary[key]))
            with open(fname, mode='a') as f:
                for row in range(0, int(np.shape(array)[0])):
                    for col in range(0, int(np.shape(array)[1])):
                        f.write(str(array[row, col])+' ')
                    f.write('\n')

        return fluxes, full_dictionary

    def load(self, fname):

        data = np.loadtxt(fname, skiprows=1)
        fluxes = data[:,0:4]
        with open(fname, 'r') as f:
            lines = f.readlines()
            header = lines[0].split(' ')
        full_dictionary = {}
        for i, key in enumerate(header):
            full_dictionary.update({key: data[:,i+4]})

        return fluxes, full_dictionary

    def _setup_data(self):

        delta_x = [np.random.normal(0, delta) for delta in self.data.sigma_x]
        delta_y = [np.random.normal(0, delta) for delta in self.data.sigma_x]

        new_m = []
        for i in range(0, len(delta_x)):
            dm = np.random.normal(0, self.data.sigma_m[i])
            new_m.append(self.data.m[i] + dm)

        delta_x, delta_y = np.array(delta_x), np.array(delta_y)

        new_m = np.array(new_m)
        new_x = np.array(self.data.x) + delta_x
        new_y = np.array(self.data.y) + delta_y

        return Data(x=new_x, y=new_y, m=new_m, t=None, source=None, sigma_x=self.data.sigma_x,
                    sigma_y=self.data.sigma_y,
                    sigma_m=self.data.sigma_m)

    def _optimize_once(self, opt_routine, args):

        run_args = {'multiplane': False, 'n_iterations': 400, 'n_particles': 80,
                    'tol_centroid': 0.05, 'verbose': False, 'tol_mag': 0.3,
                    'restart': 1, 're_optimize': False, 'optimize_routine': 'fixed_powerlaw_shear'}

        for argname in args.keys():
            run_args.update({argname: args[argname]})

        run_args.update({'datatofit': self._setup_data()})

        return opt_routine(**run_args)

    def _optimize_once_fromlensclass(self, opt_routine, args):

        run_args = {}
        for argname in args.keys():
            run_args.update({argname: args[argname]})

        run_args.update({'datatofit': self._setup_data()})
        macromodel = run_args['macromodel']
        del run_args['macromodel']

        return opt_routine(kwargs_fit = run_args, macro_init = macromodel)

