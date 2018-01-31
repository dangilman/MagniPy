import numpy as np

def integrate_profile(profname,limit,inspheres=False,**kwargs):
    if profname=='nfw':
        rs=kwargs['rs']
        ks=kwargs['ks']
        n=limit*rs**-1
        if inspheres:
            rho0 = 86802621404*ks*rs**-1
            n*=rs
            r200 = kwargs['c']*rs

            return 4*np.pi*rho0*rs**3*(np.log(1+r200*n**-1)- n*(n+r200)**-1)

        else:
            return 2*np.pi*rs**2*ks*(np.log(.25*n**2)+2*np.arctanh(np.sqrt(1-n**2))*(np.sqrt(1-n**2))**-1)
    elif profname=='SIE':
        b = kwargs['SIE_Rein']
        return np.pi*limit*b

def rotate(xcoords,ycoords,angle):
    angle*=np.pi*180**-1
    #angle+=-np.pi*.5

    return xcoords*np.cos(angle)+ycoords*np.sin(angle),-xcoords*np.sin(angle)+ycoords*np.cos(angle)

def dr(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def img_sept(x,y):

    return np.sort(np.array([dr(x[0],x[1],y[0],y[1]),dr(x[0],x[2],y[0],y[2]),dr(x[0],x[3],y[0],y[3]),
         dr(x[1],x[2],y[1],y[2]),dr(x[1],x[3],y[1],y[3]),dr(x[2],x[3],y[2],y[3])]))

def identify(x,y,RE):

    separations = img_sept(x,y)

    if separations[0]>=.7*RE:
        return 0

    else:

        if separations[1]<=1.2*RE:
            return 2
        else:
            return 1

def read_dat_file(fname):

    x_srcSIE, y_srcSIE = [], []

    with open(fname, 'r') as f:

        nextline = False
        dosrc = False
        doimg = False
        count = 0
        readcount = 0

        for line in f:
            row = line.split(" ")
            row_split = filter(None, row)
            if row_split[0] == 'alpha':
                macromodel = row_split

                continue

            if row_split[0] == 'Source':
                nextline = True
                dosrc = True
                src = []
                continue

            if nextline and dosrc:

                for item in row:
                    try:
                        src.append(float(item))
                    except ValueError:
                        continue
                x_srcSIE.append(src[0])
                y_srcSIE.append(src[1])
                nextline = False
                dosrc = False
                continue

            if row_split[0] == 'images:\n':
                nextline = True
                doimg = True
                count = 0
                x, y, f, t = [], [], [], []
                continue

            if nextline and doimg:

                count += 1
                numbers = []
                for item in row:
                    try:
                        numbers.append(float(item))
                    except ValueError:
                        continue

                x.append(numbers[4])
                y.append(numbers[5])
                f.append(numbers[6])
                t.append(numbers[7])

                if int(count) == 4:

                    t = np.array(t)

                    if min(t) < 0:
                        t += -1 * min(t)

                    xpos = x
                    ypos = y
                    fr = np.array(f)
                    tdel = np.array(t)

                    return xpos, ypos, fr, t, macromodel, [x_srcSIE[0], y_srcSIE[0]]


def read_gravlens_out(fnames):

    vector = []

    if isinstance(fnames,list):

        for fname in fnames:
            with open(fname, 'r') as f:
                lines = f.readlines()
            f.close()

            imgline = lines[1].split(' ')
            numimg = int(imgline[1])
            xpos, ypos, mag, tdelay = [], [], [], []

            for i in range(0, numimg):
                data = lines[2 + i].split(' ')
                data = filter(None, data)
                xpos.append(float(data[0]))
                ypos.append(float(data[1]))
                mag.append(np.absolute(float(data[2])))
                tdelay.append(float(data[3]))
            vector.append([np.array(xpos), np.array(ypos), np.array(mag), np.array(tdelay), numimg])
    else:
        with open(fnames, 'r') as f:
            lines = f.readlines()
        f.close()

        imgline = lines[1].split(' ')
        numimg = int(imgline[1])
        xpos, ypos, mag, tdelay = [], [], [], []

        for i in range(0, numimg):
            data = lines[2 + i].split(' ')
            data = filter(None, data)
            xpos.append(float(data[0]))
            ypos.append(float(data[1]))
            mag.append(np.absolute(float(data[2])))
            tdelay.append(float(data[3]))
        vector.append([np.array(xpos), np.array(ypos), np.array(mag), np.array(tdelay), numimg])

    return vector


def read_chain_out(fname, N=1):
    nimg, srcx, srcy, x1, y1, m1, t1, x2, y2, m2, t2, x3, y3, m3, t3, x4, y4, m4, t4 = np.loadtxt(fname, unpack=True)

    return nimg, [srcx, srcy], [x1, x2, x3, x4], [y1, y2, y3, y4], [m1, m2, m3, m4], [t1, t2, t3, t4]


def polar_to_cart(ellip, theta, polar_to_cart = True):

    xcomp = ellip*np.cos(2*theta*np.pi*180**-1)
    ycomp = ellip*np.sin(2*theta*np.pi*180**-1)
    return xcomp,ycomp

def cart_to_polar(e1, e2, polar_to_cart = True):

    if e1==0:
        return 0,0
    else:
        return np.sqrt(e1**2+e2**2),0.5*np.arctan2(e2,e1)*180*np.pi**-1

def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(len(array)))
        if n**2 != len(array):
            raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh

def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """

    :param numPix: number of pixels per axis
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if left_lower is True:
        x_grid = matrix[:, 0]*deltapix
        y_grid = matrix[:, 1]*deltapix
    else:
        x_grid = (matrix[:, 0] - (numPix_eff-1)/2.)*deltapix_eff
        y_grid = (matrix[:, 1] - (numPix_eff-1)/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    return array2image(x_grid - shift), array2image(y_grid - shift)


