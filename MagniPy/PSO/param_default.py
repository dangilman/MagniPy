class ParamDefaultRanges(object):

    def get_ranges(self,kwargs):

        ranges = []

        for key,item in kwargs.iteritems():

            ranges.append(self._get_ranges(key,item))

        return ranges

    def _get_ranges(self,pname,guess):

        if pname == 'theta_E':
            width = 0.2
            return [guess-width,guess+width]

        if pname in ['e1','e2']:
            width = 0.025
            return [guess-width,guess+width]

        if pname in ['center_x','center_y']:
            width = 0.01
            return [guess-width,guess+width]
