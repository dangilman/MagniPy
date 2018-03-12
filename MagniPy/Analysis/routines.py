import numpy as np

def asimuthal_avg(two_D_array):

    N = two_D_array.shape[0]

    bins = N*10**-1

    xx,yy = np.meshgrid(np.linspace(-N*.5,N*.5,bins),np.linspace(-N*.5,N*.5,bins))
    r = (xx**2+yy**2)**.5
    dr = N*bins**-1
    p = []
    for i in range(1,int(bins)):

        inds = np.where(r<dr*i)
        p.append(np.sum(two_D_array[inds])*len(inds)**-2)
    return p

