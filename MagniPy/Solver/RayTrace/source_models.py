import numpy as np
import matplotlib.pyplot as plt

class GAUSSIAN:

    def __init__(self,x,y,width):

        self.xcenter,self.ycenter,self.width = x,y,width

    def source_profile(self,betax,betay):

        dx,dy = -self.xcenter + betax,-self.ycenter + betay

        return (2*np.pi*self.width**2)**-1*np.exp(-0.5*(dx**2+dy**2)*self.width**-2)