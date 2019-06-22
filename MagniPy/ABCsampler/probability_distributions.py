import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def approx_cdf_1d(x_array, pdf_array):
    """

    :param x_array: x-values of pdf
    :param pdf_array: pdf array of given x-values
    """
    norm_pdf = pdf_array / np.sum(pdf_array)
    cdf_array = np.zeros_like(norm_pdf)
    cdf_array[0] = norm_pdf[0]
    for i in range(1, len(norm_pdf)):
        cdf_array[i] = cdf_array[i - 1] + norm_pdf[i]
    cdf_func = interp1d(x_array, cdf_array)
    cdf_inv_func = interp1d(cdf_array, x_array)
    return cdf_array, cdf_func, cdf_inv_func

class ProbabilityDistribution(object):


    def __init__(self,distribution_type='',args={},Nsamples=int,decimals=int,**kwargs):

        self.decimals = decimals

        if distribution_type=='Uniform':
            self.draw  = self.Uniform
            self.low,self.high = args['low'],args['high']

        elif distribution_type=='LogUniform':
            self.draw  = self.LogUniform
            self.low,self.high = args['low'],args['high']

        elif distribution_type=='Gaussian':
            self.draw = self.Gaussian
            self.mean,self.sigma = args['mean'],args['sigma']
            self.positive_definite = args['positive_definite']

        elif distribution_type=='PDF':

            self.draw = self.InvertCDF

            sorted = np.argsort(args['values'])
            args['values'] = np.array(args['values'])
            args['pdf'] = np.array(args['pdf'])
            values = args['values'][sorted]
            pdf = args['pdf'][sorted]
            norm = np.max(pdf)

            self.values, self.pdf = values, pdf/norm

        else:
            raise Exception('distribution_type not recognized: ')

        if 'sort_ascending' in args:
            self.sort_ascending = True
        else:
            self.sort_ascending = False

    def InvertCDF(self, N):

        zmin, zmax = self.values[0], self.values[-1]

        pz = interp1d(self.values, self.pdf)
        samples = []
        while len(samples)<N:
            zprop = np.random.uniform(zmin, zmax)
            urand = np.random.uniform(0, 1)

            if pz(zprop) > urand:
                samples.append(zprop)

        return np.round(samples, self.decimals)

    def LogUniform(self, N):

        logsamples = np.random.uniform(self.low, self.high, N)
        if self.sort_ascending:
            logsamples = logsamples[np.argsort(logsamples)]

        samples = 10**logsamples

        return np.round(np.array(samples), self.decimals)

    def Uniform(self,N):

        samples = np.random.uniform(self.low,self.high,N)

        if self.sort_ascending:
            samples = samples[np.argsort(samples)]

        return np.round(np.array(samples),self.decimals)


    def Gaussian(self,N):

        samples = np.random.normal(self.mean,self.sigma,N)

        if self.positive_definite:
            samples = np.absolute(samples)

        if self.sort_ascending:
            samples = samples[np.argsort(samples)]

        return np.round(np.array(samples),self.decimals)

