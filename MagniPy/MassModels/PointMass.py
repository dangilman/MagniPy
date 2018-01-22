from MagniPy.LensBuild.cosmology import Cosmo

class PointMass(Cosmo):

    #pcrit = 2.77536627e+11

    def __init__(self,z1=0.5,z2=1.5,h=0.7,c_turnover=True):
        """
        adopting a standard cosmology, other cosmologies not yet implemented
        :param z1: lens redshift
        :param z2: source redshift
        :param h: little h
        """
        Cosmo.__init__(self, z1=z1, z2=z2)
        self.c_turnover=c_turnover

    def params(self,M):
        subkwargs = {}
        subkwargs['b'] = self.R_ein(M)
        return subkwargs

    def R_ein(self,M):
        #print self.D_ds*self.D_d**-1*self.D_s**-1
        return (M*self.G*self.c**-2*self.kpc_per_asec(self.zd)**-1*self.D_ds*self.D_d**-1*self.D_s**-1*1000)**.5

p = PointMass()

print 1000*p.R_ein(10**7)