import numpy as np
from copy import deepcopy

class ProbabilityDistribution:

    def __init__(self,distribution_type='',args={},Nsamples=int,decimals=int):

        self.decimals = decimals

        if distribution_type=='Uniform':
            self.draw  = self.Uniform
            self.low,self.high = args['low'],args['high']
            self.steps = args['steps']

        elif distribution_type=='Gaussian':
            self.draw = self.Gaussian
            self.mean,self.sigma = args['mean'],args['sigma']
            self.positive_definite = args['positive_definite']

        else:
            raise Exception('distribution_type not recognized: ')

        if 'sort_ascending' in args:
            self.sort_ascending = True
        else:
            self.sort_ascending = False

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
