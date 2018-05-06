import numpy as np
import matplotlib.pyplot as plt

class GAUSSIAN:

    def __init__(self,x,y,width,xgrid0,ygrid0):

        self.xcenter,self.ycenter,self.width = x,y,width

    def __call__(self,betax,betay):

        dx,dy = -self.xcenter + np.array(betax),-self.ycenter + np.array(betay)

        return (2*np.pi*self.width**2)**-1*np.exp(-0.5*(dx**2+dy**2)*self.width**-2)