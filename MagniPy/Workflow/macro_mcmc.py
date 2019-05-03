import numpy as np
from MagniPy.lensdata import Data
from MagniPy.LensBuild.defaults import get_default_SIE
from copy import deepcopy

class MacroMCMC(object):

    def __init__(self, lensdata_class, optroutine):

        self.data = lensdata_class
        self.optroutine = optroutine

    def run(self, macromodel=None,
            tovary={'source_size_kpc': [0.01, 0.05], 'gamma': [1.95, 2.2]}, N=500,
            zlens=0.5, optimizer_kwargs={}, write_to_file=False, fname=None):

        if write_to_file is True:
            assert fname is not None

        rein, ellip, PA, shear, shearPA, gamma, source_size, cen_x, cen_y = \
            [], [], [], [], [], [], [], [], []

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

            if 'gamma' in tovary.keys():
                gammavalue = np.random.uniform(tovary['gamma'][0], tovary['gamma'][1])
                macro.lenstronomy_args['gamma'] = gammavalue
            if 'source_size_kpc' in tovary.keys():
                srcsize = np.random.uniform(tovary['source_size_kpc'][0],
                                            tovary['source_size_kpc'][1])
                args.update({'source_size_kpc': srcsize})

            args.update({'macromodel': macro})

            out = self._optimize_once(self.optroutine, args)

            optdata, optmodel = out[0][0], out[1][0]

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
                fluxes = optdata.m
            else:
                fluxes = np.vstack((fluxes, optdata.m))

        rein = np.array(rein)
        ellip = np.array(ellip)
        PA = np.array(PA)
        shear = np.array(shear)
        shearPA = np.array(shearPA)
        gamma = np.array(gamma)
        source_size = np.array(source_size)
        cen_x, cen_y = np.array(cen_x), np.array(cen_y)
        full_dictionary = {'Rein': rein, 'ellip': ellip, 'ellipPA': PA,
                           'shear': shear, 'shearPA': shearPA, 'gamma': gamma,
                           'srcsize': source_size, 'centroid_x': cen_x, 'centroid_y': cen_y}

        if write_to_file:
            array = fluxes
            header = 'f1 f2 f3 '
            for key in full_dictionary.keys():
                header += key + ' '
                array = np.column_stack((array, full_dictionary[key]))

            np.savetxt(fname, X=array, header=header)

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

        run_args = {'multiplane': False, 'n_iterations': 400, 'n_particles': 50,
                    'tol_centroid': 0.05, 'tol_mag': [1] * 4, 'verbose': False,
                    'restart': 1, 're_optimize': False, 'optimize_routine': 'fixed_powerlaw_shear'}

        for argname in args.keys():
            run_args.update({argname: args[argname]})

        run_args.update({'datatofit': self._setup_data()})

        return opt_routine(**run_args)

