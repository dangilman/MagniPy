import numpy as np

def truncation(r3d, RE, m, sigmacrit):
    return (m * r3d ** 2 * (2 * sigmacrit * RE) ** -1) ** (1. / 3)

class Uniform:

    def __init__(self,Rmax):

        self.Rmax = Rmax
        self.area = np.pi*Rmax**2

    def draw(self,N):

        r3d, x, y = Runi(int(round(N)), self.Rmax)

        return np.array(r3d), np.array(x), np.array(y)


class Uniform_2d_1:

    def __init__(self,Rmax2d,Rmaxz,rc):

        self.Rmax2d = Rmax2d
        self.Rmaxz = Rmaxz
        self.rc = rc
        self.area = np.pi*Rmax2d**2

    def r3d_pdf(self,r):
        def f(x):
            return np.arcsinh(x)-x*(x**2+1)**-.5

        norm = (4*np.pi*self.rc**3*f(self.Rmaxz*self.rc**-1))**-1
        return norm*(1+r**2*self.rc**-2)**-1.5

    def draw(self,N):

        def acceptance_prob(r):

            return self.r3d_pdf(r)*self.r3d_pdf(0)**-1

        r3d,x,y = Runi(N,self.Rmax2d,Rmaxz=self.Rmaxz)

        for i in range(0,int(len(r3d))):
            u = np.random.rand()

            accept = acceptance_prob(r3d[i])

            while u>=accept:
                r3d[i],x[i],y[i] = Runi(1,self.Rmax2d,Rmaxz=self.Rmaxz)
                u = np.random.rand()
                accept = acceptance_prob(r3d[i])

        return r3d,x,y

class Uniform_2d_nfw:

    def __init__(self,Rmax2d,Rmaxz,rs,core_factor=0.3):

        self.Rmax2d = Rmax2d
        self.Rmaxz = Rmaxz
        self.rs = rs
        self.area = np.pi*Rmax2d**2
        self.corefactor=0.3

    def r3d_pdf(self,r,rtol=1):
        r+=1e-9
        return ((r*self.rs**-1)*(1+r*self.rs**-1)**2)**-1

    def r2d_pdf(self,r,rtol=1):
        def F(x):

            if isinstance(x, np.ndarray):
                nfwvals = np.ones_like(x)
                inds1 = np.where(x < 1)
                inds2 = np.where(x > 1)
                nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
                nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
                return nfwvals

            elif isinstance(x, float) or isinstance(x, int):
                if x == 1:
                    return 1
                if x < 1:
                    return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
                else:
                    return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)
        return (1-F(r*self.rs**-1))*(r**2**self.rs**-2-1)**-1

    def draw(self,N,near_x=False,near_y=False,mindis=False):

        def acceptance_prob(r):

            return self.r3d_pdf(r)*self.r3d_pdf(self.rs*self.corefactor)**-1

        r3d,x,y = Runi(N,self.Rmax2d,Rmaxz=self.Rmaxz)

        for i in range(0,int(len(r3d))):
            u = np.random.rand()

            accept = acceptance_prob(r3d[i])


            while u>=accept:
                r3d[i],x[i],y[i] = Runi(1,self.Rmax2d,Rmaxz=self.Rmaxz)
                u = np.random.rand()
                accept = acceptance_prob(r3d[i])

        return r3d,x,y

def Runi(N, Rmax, Rmaxz = None):
    if Rmaxz is None:
        Rmaxz = Rmax
    xrand = Rmax * np.random.uniform(-1, 1, N)
    yrand = Rmax * np.random.uniform(-1, 1, N)
    zrand = Rmaxz * np.random.uniform(-1, 1, N)
    R_3D = np.sqrt(xrand**2+yrand**2+zrand**2)
    for i in range(0, np.shape(R_3D)[0]):
        R_new = np.sqrt(float(xrand[i]) ** 2 + float(yrand[i]) ** 2)
        while R_new > Rmax:
            xrand[i], yrand[i] = Rmax * np.random.uniform(-1, 1, 1),Rmax * np.random.uniform(-1, 1, 1)
            R_new = np.sqrt(float(xrand[i]) ** 2 + float(yrand[i]) ** 2)

    return np.sqrt(xrand ** 2 + yrand ** 2 + zrand ** 2), xrand, yrand


def filter_spatial_2(xsub, ysub, r3d, xpos, ypos, masses, mindis, between_low, between_high, Nsub):
    """
    same as filter spatial, except it will exclude subhalos between_low < M < between_high
    :param xsub: sub x coords
    :param ysub: sub y coords
    :param xpos: img x coords
    :param ypos: img y coords
    :param mindis: max 2d distance
    :return: filtered subhalos
    """


    if Nsub==1:
        keep=False
        for i in range(0,len(xpos)):

            r =np.sqrt((xsub[0]-xpos[i])**2+(ysub[0]-ypos[i])**2)

            if masses[0] >= between_high or masses[0] <= between_low:
                if masses[0]>10**8 or r<mindis:

                    keep=True
                    break

        if keep:
            return [0]
        else:
            return []
    else:
        inds = []

        for j in range(0,Nsub-1):
            for i in range(0,len(xpos)):
                r = np.sqrt((xsub[j] - xpos[i]) ** 2 + (ysub[j] - ypos[i]) ** 2)
                if masses[j] >= between_high or masses[j] <= between_low:
                    if r<mindis or masses[j]>10**8:

                        inds.append(j)
                        break

        return inds



