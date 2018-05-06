import numpy as np

class Sersic:

    def b(self,n):
        return 1.9992*n - 0.3271 + 4*(405*n)**-1

    def kappa(self,x, y, n_sersic, r_eff, k_eff, q, center_x=0, center_y=0):

        bn = self.b(n_sersic)

        r = (x**2+y**2*q**-2)**0.5

        return k_eff*np.exp(-bn*((r*r_eff**-1)**(n_sersic**-1)-1))
