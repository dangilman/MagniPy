import numpy as np

class Gaussian1d:

    def __init__(self, mean, weight, width):

        self.mean = mean
        self.weight = weight
        self.sigma = width

    def __call__(self, x):

        exponent = -0.5*(x - self.mean) ** 2 * self.sigma ** -2

        return self.weight * np.exp(exponent)

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


class Boundary2D(object):

    def __init__(self,scale=3,size=100):
        self.size = size
        self.scale = scale

    def near_boundary(self, x, dx, ranges):

        low, high = ranges[0], ranges[1]

        if x < low:
            return False
        if x > high:
            return False

        close_to_xlow = (x - low) * dx**-1
        close_to_xhigh = (high - x) * dx ** -1

        if close_to_xlow < 1 or close_to_xhigh < 1:
            return True
        else:
            return False

    def renormalize(self,xi,yi,kx,ky,pranges):

        cov = [[(kx) ** 2, 0], [0, (ky) ** 2]]

        near_x_boundary = self.near_boundary(xi, kx, pranges[0])
        near_y_boundary = self.near_boundary(yi, ky, pranges[1])

        condition1 = np.logical_and(near_x_boundary, near_y_boundary)

        if condition1:
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
            xsamp, ysamp = samples[:, 0], samples[:, 1]
            inside_y = np.logical_and(ysamp >= pranges[1][0], ysamp <= pranges[1][1])
            inside_x = np.logical_and(xsamp >= pranges[0][0], xsamp <= pranges[0][1])
            inside_inds = float(np.sum(np.logical_and(inside_x, inside_y)))

        elif near_x_boundary:
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
            xsamp = samples[:, 0]
            inside_x = np.logical_and(xsamp >= pranges[0][0], xsamp <= pranges[0][1])
            inside_inds = float(np.sum(inside_x))

        elif near_y_boundary:
            samples = np.random.multivariate_normal([xi, yi], cov=cov, size=self.size)
            ysamp = samples[:, 1]
            inside_y = np.logical_and(ysamp >= pranges[1][0], ysamp <= pranges[1][1])
            inside_inds = float(np.sum(inside_y))

        else:
            return 1

        try:
            weight = self.size * inside_inds ** -1
        except:
            print(pranges[0], pranges[1])
            print(xi, yi)
            print(kx, ky)
            a=input('continue')
            weight = 1

        return weight


class Boundary1D(object):

    def __init__(self,scale=3,size=50):
        self.size = size
        self.scale = scale

    def near_boundary(self, x, dx, ranges):

        low, high = ranges[0], ranges[1]

        if x < low:
            return False
        if x > high:
            return False

        close_to_xlow = (x - low) * dx**-1
        close_to_xhigh = (high - x) * dx ** -1

        if close_to_xlow < 1 or close_to_xhigh < 1:
            return True
        else:
            return False

    def renormalize(self,xi,kx,pranges):

        if self.near_boundary(xi, kx, pranges[0]):
            samples = np.random.normal(xi, kx, size=self.size)
        else:
            return 1

        inside_x = np.sum(np.logical_and(samples>pranges[0][0],samples<pranges[0][1]))

        return self.size * float(inside_x)**-1



