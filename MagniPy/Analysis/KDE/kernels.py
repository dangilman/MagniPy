import numpy as np

class Gaussian2d:

    def __init__(self, meanx, meany, weight,covmat=None):

        self.meanx = meanx
        self.meany = meany

        self.weight = weight
        self.covmat = covmat

    def __call__(self, x, y):

        vec = np.array([x-self.meanx,y-self.meany])

        dr2 = np.dot(vec.T,np.dot(vec,self.covmat))

        exponent = -0.5 * dr2

        return self.weight*np.exp(exponent)

class Silverman2d:

    def __init__(self, meanx, meany, weight,covmat=None):

        self.meanx = meanx
        self.meany = meany

        self.weight = weight
        self.covmat = covmat

    def __call__(self, x, y):

        vec = np.array([x-self.meanx,y-self.meany])

        dr = np.dot(vec.T,np.dot(vec,self.covmat))**0.5

        root2 = 2**0.5
        exponent = -root2**-1 * dr

        return 0.5*self.weight*np.exp(exponent)*np.sin(dr*root2**-1 + np.pi*0.25)

class Epanechnikov:

    def __init__(self, meanx, meany, weight,covmat=None):

        self.meanx = meanx
        self.meany = meany

        self.weight = weight
        self.covmat = covmat

    def __call__(self, x, y):

        vec = np.array([x-self.meanx,y-self.meany])

        dr2 = np.dot(vec.T,np.dot(vec,self.covmat))

        if dr2 > 1:
            return 0
        return 0.75*(1-dr2)

class Sigmoid:

    def __init__(self, meanx, meany, weight,covmat=None):

        self.meanx = meanx
        self.meany = meany

        self.weight = weight
        self.covmat = covmat

    def __call__(self, x, y):

        vec = np.array([x-self.meanx,y-self.meany])

        dr = np.dot(vec.T,np.dot(vec,self.covmat))**0.5

        return 2*np.pi**-1*(np.exp(dr)+np.exp(-dr))**-1


class Boundary:

    def __init__(self,scale=2,size=400):
        self.size = size
        self.scale = scale

    def outside_boundary(self,x,dx,low,high):

        close_to_boundary = np.logical_or(x + dx < low, x +dx > high)

        if close_to_boundary is False:
            return False

        if x < low-dx:
            return False
        if x > high + dx:
            return False

    def renormalize(self,xi,yi,kx,ky,pranges):

        cov = [[(kx) ** 2, 0], [0, (ky) ** 2]]

        if self.outside_boundary(xi,-kx*self.scale,pranges[0][0],pranges[0][1]):
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
        elif self.outside_boundary(xi,kx*self.scale,pranges[0][0],pranges[0][1]):
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
        elif self.outside_boundary(yi,-ky*self.scale,pranges[1][0],pranges[1][1]):
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
        elif self.outside_boundary(yi,ky*self.scale,pranges[1][0],pranges[1][1]):
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
        else:
            return 1

        xsamp,ysamp = samples[:,0],samples[:,1]

        inside_y = np.logical_and(ysamp>pranges[1][0],ysamp<pranges[1][1])
        inside_x = np.logical_and(xsamp>pranges[0][0],xsamp<pranges[0][1])

        inside_inds = np.logical_and(inside_x,inside_y)

        return self.size*float(np.sum(inside_inds)+1)**-1



