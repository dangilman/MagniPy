from MagniPy.util import polar_to_cart
from param_default import ParamDefaultRanges
import numpy as np

class Params(object):

    def __init__(self,zlist, lens_list, arg_list,init_limits=None,optimizer_routine='optimize_SIE_SHEAR',fixed_params=None):

        self.N_to_vary = self.get_vary_length(optimizer_routine)

        self.model_fixed = self.get_model_fixed(optimizer_routine, fixed_params)

        self.zlist_tovary = zlist[0:self.N_to_vary]
        self.lenslist_tovary = lens_list[0:self.N_to_vary]
        self.args_tovary = arg_list[0:self.N_to_vary]

        self.zlist_fixed = zlist[self.N_to_vary:]
        self.lenslist_fixed = lens_list[self.N_to_vary:]

        self.args_fixed = arg_list[self.N_to_vary:]

        self.tovary_lower_limit,self.tovary_upper_limit = self.get_bounds(init_limits)

        self.Pbounds = ParamDefaultRanges()

    def argstovary_values(self):

        arg_array = []

        for key in self.args_tovary.keys():

            if key not in self.model_fixed.keys():
                arg_array.append(self.args_tovary[key])
            else:
                arg_array.append(self.model_fixed[key])


        return np.array(arg_array)

    def argsfixed_values(self):

        return np.array([args.values() for args in self.args_fixed])

    def argstovary_todictionary(self,values):

        args_list = []
        count = 0

        for n in range(0,self.N_to_vary):

            args = {}

            for key in self.args_tovary[n].keys():

                if key in self.model_fixed:
                    args.update({key:self.model_fixed[key]})
                else:
                    args.update({key:values[count]})
                    count += 1

            args_list.append(args)

        return args_list

    def argsfixed_todictionary(self):

        return self.args_fixed

    def get_bounds(self,init_limits):

        tovary_low = []
        tovary_high = []

        if init_limits is not None:
            return self.Pbounds.get_ranges(init_limits)

        for lens_name in self.lenslist_tovary:

            lo,hi = self._pbounds(lens_name)

            tovary_low += lo
            tovary_high += hi

        return tovary_low,tovary_high

    def _pbounds(self,pname):

        low_e12 = -0.01
        hi_e12 = 0.01

        if pname == 'SPEMD':

            low_Rein = 0.7
            hi_Rein = 1.4

            low_center = -0.01
            hi_center = 0.01

            return [low_Rein,low_center,low_center,low_e12,low_e12],\
                   [hi_Rein,hi_center,hi_center,hi_e12,hi_e12]

        elif pname == 'SHEAR':

            return [low_e12,low_e12],[hi_e12,hi_e12]

    def get_vary_length(self,routine):

        if routine in ['optimize_SIE_SHEAR','optimize_plaw_shear']:

            return 2

    def get_model_fixed(self,routine,fixed_params):

        if routine == 'optimize_SIE_SHEAR':

            return {'gamma':2}

        elif routine == 'optimize_plaw_shear':

            return {'gamma':fixed_params['gamma']}





