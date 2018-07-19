from MagniPy.util import *
import numpy as np
from copy import copy

class Params(object):

    def __init__(self,zlist=None, lens_list=None, arg_list=None,
                 optimizer_routine='optimize_SIE_SHEAR',fixed_params=None, initialized=True):

        Ntovary,self.vary_inds = self.get_vary_length(optimizer_routine)

        self.model_fixed = self.get_model_fixed(optimizer_routine, fixed_params)

        self.Ntovary = Ntovary

        if Ntovary == 2:

            self.zlist_tovary = zlist[self.vary_inds[0]:self.vary_inds[1]]
            self.lenslist_tovary = lens_list[self.vary_inds[0]:self.vary_inds[1]]
            self.args_tovary = arg_list[self.vary_inds[0]:self.vary_inds[1]]

            self.zlist_fixed = zlist[self.vary_inds[1]:]
            self.lenslist_fixed = lens_list[self.vary_inds[1]:]
            self.args_fixed = arg_list[self.vary_inds[1]:]

        elif Ntovary == 1:

            self.zlist_tovary = [zlist[self.vary_inds]]
            self.lenslist_tovary = [lens_list[self.vary_inds]]
            self.args_tovary = [arg_list[self.vary_inds]]

            _z = copy(zlist)
            _l = copy(lens_list)
            _l2 = copy(arg_list)
            del _z[self.vary_inds]
            del _l[self.vary_inds]
            del _l2[self.vary_inds]

            self.zlist_fixed = [_z]
            self.lenslist_fixed = [_l]
            self.args_fixed = [_l2]

        self.Pbounds = ParamRanges()

        if initialized is False:

            self.tovary_lower_limit,self.tovary_upper_limit = self.Pbounds.get_ranges(self.lenslist_tovary,None,optimizer_routine)
        else:

            self.tovary_lower_limit, self.tovary_upper_limit = self.Pbounds.get_ranges(self.lenslist_tovary,self.args_tovary,optimizer_routine)

        self.scale = self.Pbounds.get_scale(optimizer_routine)

    def argsfixed_values(self):

        return np.array([args.values() for args in self.args_fixed])

    def argstovary_todictionary(self,values):

        args_list = []
        count = 0

        for n in range(0, int(self.Ntovary)):

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

    def get_model_fixed(self, routine, fixed_params):

        if routine == 'optimize_SIE_shear':

            return {'gamma': 2}

        elif routine == 'optimize_plaw_shear':

            return {'gamma': fixed_params['gamma']}

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

    def get_vary_length(self,routine):

        if routine in ['optimize_SIE_shear','optimize_plaw_shear']:

            return 2,[0,2]

        elif routine in ['optimize_SHEAR']:

            return 1,1

    def get_model_fixed(self,routine,fixed_params):

        if routine == 'optimize_SIE_shear':

            return {'gamma':2}

        elif routine == 'optimize_plaw_shear':

            return {'gamma':fixed_params['gamma']}

        elif routine == 'optimze_SHEAR':

            return {}

class ParamRanges(object):

    def get_scale(self,routine):

        if routine == 'optimize_SIE_shear':

            return [0.01,0.01,0.01,0.01,0.01,0.01,0.01]

        else:

            raise Exception('routine not recognized.')

    def get_ranges(self,lens_names,args_init,routine):

        ranges_low,ranges_high = [],[]

        for idx,lens in enumerate(lens_names):

            if routine in ['optimize_SIE_shear','optimize_plaw_shear']:
                if args_init is None:

                    _ranges_low,_ranges_high = self._get_ranges(lens)

                else:
                    _ranges_low,_ranges_high = [],[]
                    if lens == 'SPEMD':
                        _ranges_low_, _ranges_high_ = self._get_ranges_initialized(lens, args_init[0])
                    elif lens == 'SHEAR':
                        _ranges_low_,_ranges_high_ = self._get_ranges_initialized(lens, args_init[1])
                    _ranges_low += _ranges_low_
                    _ranges_high += _ranges_high_

                ranges_low += _ranges_low
                ranges_high += _ranges_high

            elif routine in ['optimize_SHEAR']:

                if args_init is None:

                    _ranges_low,_ranges_high = self._get_ranges(lens)

                else:
                    _ranges_low, _ranges_high = [], []
                    assert lens == 'SHEAR'

                    _ranges_low_, _ranges_high_ = self._get_ranges_initialized(lens, args_init[0])
                    _ranges_low += _ranges_low_
                    _ranges_high += _ranges_high_

                ranges_low += _ranges_low
                ranges_high += _ranges_high

        return ranges_low,ranges_high

    def _get_ranges(self,lens_name):

        low_e12 = -0.1
        hi_e12 = 0.1

        low_shear_e12 = 0.05
        high_shear_e12 = 0.05

        if lens_name == 'SPEMD':

            low_Rein = 0.7
            hi_Rein = 1.4

            low_center = -0.01
            hi_center = 0.01

            return [low_Rein,low_center,low_center,low_e12,low_e12],\
                   [hi_Rein,hi_center,hi_center,hi_e12,hi_e12]

        elif lens_name == 'SHEAR':

            return [low_shear_e12,low_shear_e12],[high_shear_e12,high_shear_e12]

    def _get_ranges_initialized(self, lens_name, args_init):

        ranges_low,ranges_high = [],[]

        if lens_name == 'SPEMD':

            for pname,guess in args_init.iteritems():

                if pname == 'theta_E':
                    width = 0.005
                    ranges_low += [guess-width]
                    ranges_high += [guess+width]

                if pname in ['e1','e2']:

                    width = 0.025

                    ranges_low += [guess - width]
                    ranges_high += [guess + width]

                if pname in ['center_x','center_y']:
                    width = 0.0025
                    ranges_low += [guess-width]
                    ranges_high += [guess+width]

            return ranges_low,ranges_high

        elif lens_name == 'SHEAR':

            width_shear,width_shear_theta = 0.025,30
            shear,shear_theta = cart_to_polar(args_init['e1'],args_init['e2'])

            shear_low = max(1e-5,shear-width_shear)
            shear_theta_low = shear_theta-width_shear_theta

            shear_high = shear+width_shear
            shear_theta_high = shear_theta+width_shear_theta

            e1_low,e2_low = polar_to_cart(shear_low,shear_theta_low)
            e1_high, e2_high = cart_to_polar(shear_high, shear_theta_high)


            return [e1_low,e2_low],[e1_high,e2_high]





