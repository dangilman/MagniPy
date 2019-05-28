import numpy as np
from MagniPy.lensdata import Data
import subprocess
import shutil
import scipy.ndimage.filters as sfilt
import itertools
from copy import deepcopy

def snap_to_bins(data, xbin_centers, dx, ybin_centers, dy, ranges):

    new_datax = deepcopy(data[:, 0])
    new_datay = deepcopy(data[:, 1])
    new_datax[np.where(new_datax <= ranges[0][0])] = xbin_centers[0]
    new_datax[np.where(new_datax >= ranges[0][1])] = xbin_centers[-1]

    new_datay[np.where(new_datay <= ranges[1][0])] = ybin_centers[0]
    new_datay[np.where(new_datay >= ranges[1][1])] = ybin_centers[-1]

    new_data = None
    xx, yy = np.meshgrid(xbin_centers, ybin_centers)
    coords = zip(np.round(xx.ravel(), 4), np.round(yy.ravel(), 4))

    for i, (cenx, ceny) in enumerate(coords):

        subx = np.absolute(new_datax - cenx) * dx ** -1
        suby = np.absolute(new_datay - ceny) * dy ** -1
        inds = np.where(np.logical_and(subx < 1, suby < 1))[0]
        if len(inds) > 0:
            new_array = np.column_stack((np.array([cenx] * len(inds)), np.array([ceny] * len(inds))))
            if new_data is None:
                new_data = deepcopy(new_array)
            else:
                new_data = np.vstack((new_data, new_array))

    return new_data

def approx_theta_E(ximg,yimg):

    dis = []
    xinds,yinds = [0,0,0,1,1,2],[1,2,3,2,3,3]

    for (i,j) in zip(xinds,yinds):

        dx,dy = ximg[i] - ximg[j], yimg[i] - yimg[j]
        dr = (dx**2+dy**2)**0.5
        dis.append(dr)
    dis = np.array(dis)

    greatest = np.argmax(dis)
    dr_greatest = dis[greatest]
    dis[greatest] = 0

    second_greatest = np.argmax(dis)
    dr_second = dis[second_greatest]

    return 0.5*(dr_greatest*dr_second)**0.5

def min_img_sep(ximg,yimg):

    assert len(ximg) == len(yimg)
    dr = []

    if len(ximg) == 1:
        return 1
    elif len(ximg) == 0:
        return 1
    try:
        for i in range(0,int(len(ximg)-1)):
            for j in range(i+1,int(len(ximg))):
                dx = ximg[i] - ximg[j]
                dy = yimg[i] - yimg[j]
                dr.append((dx**2 + dy**2)**0.5)

        return min(dr)
    except:
        print('problem with the fit...')
        return 1

def sort_image_index(ximg,yimg,xref,yref):

    assert len(xref) == len(ximg)

    x_self = np.array(list(itertools.permutations(ximg)))
    y_self = np.array(list(itertools.permutations(yimg)))

    indexes = [0, 1, 2, 3]
    index_iterations = list(itertools.permutations(indexes))
    delta_r = []

    for i in range(0, int(len(x_self))):
        dr = 0
        for j in range(0, int(len(x_self[0]))):
            dr += (x_self[i][j] - xref[j]) ** 2 + (y_self[i][j] - yref[j]) ** 2

        delta_r.append(dr ** .5)

    min_indexes = np.array(index_iterations[np.argmin(delta_r)])

    return min_indexes

def coordinates_inbox(box_dx,box_dy,centered_x,centered_y):

    return np.logical_and(np.logical_and(-0.5*box_dx < centered_x, centered_x <  0.5*box_dx),
                          np.logical_and(-0.5*box_dy < centered_y, centered_y <  0.5*box_dy))

def confidence_interval(percentile,data):

    data=np.array(data)

    data.sort()

    L = len(data)
    counter = 0

    while True:

        value = data[counter]

        if counter>=L*percentile:
            break
        counter+=1

    return value

def quick_confidence(centers, heights, percentile):

    total = np.sum(heights)
    summ, index = 0, 0
    while summ < total * percentile:
        summ += heights[index]
        index += 1
    return centers[index-1]

def read_data(filename='',N=None):

    with open(filename,'r') as f:
        lines = f.readlines()

    dsets = []

    for line in lines:

        line = line.split(' ')

        n = int(line[0])

        try:
            srcx,srcy = float(line[1]),float(line[2])
        except:
            srcx,srcy = None,None

        x1,x2,x3,x4,y1,y2,y3,y4 = float(line[3]),float(line[7]),float(line[11]),float(line[15]),float(line[4]),\
                                  float(line[8]),float(line[12]),float(line[16])
        m1,m2,m3,m4 = float(line[5]),float(line[9]),float(line[13]),float(line[17])
        t1,t2,t3,t4 = float(line[6]),float(line[10]),float(line[14]),float(line[18])
        dsets.append(Data(x=[x1,x2,x3,x4],y=[y1,y2,y3,y4],m=[m1,m2,m3,m4],
                    t=[t1,t2,t3,t4],source=[srcx,srcy]))
        if N is not None and len(dsets)>=N:
            break

    return dsets


def write_fluxes(filename='',fluxes = [], mode='append',summed_in_quad=True):

    if summed_in_quad:
        fluxes = np.squeeze(fluxes)
        with open(filename,'a') as f:
            if isinstance(fluxes,float):
                f.write(str(fluxes)+'\n')
            else:
                for val in fluxes:
                    f.write(str(val)+'\n')
        return

    fluxes = np.array(fluxes)

    if mode == 'append':
        m = 'a'
    else:
        m = 'w'

    if fluxes.ndim == 1:

        with open(filename, m) as f:

            for val in fluxes:
                f.write(str(val) + ' ')
            f.write('\n')
    else:

        N = int(np.shape(fluxes)[0])

        with open(filename,m) as f:
            for n in range(0,N):
                for val in fluxes[n,:]:
                    f.write(str(val)+' ')
                f.write('\n')

def write_data(filename='',data_list=[],mode='append'):

    def single_line(dset=classmethod):
        lines = ''
        lines += str(dset.nimg)+' '+str(dset.srcx)+' '+str(dset.srcy)+' '

        for i in range(0,int(dset.nimg)):

            for value in [dset.x[i],dset.y[i],dset.m[i],dset.t[i]]:
                if value is None:
                    lines += '0 '
                else:
                    lines += str(value)+' '

        return lines+'\n'

    if mode=='append':
        with open(filename,'a') as f:
            for dataset in data_list:

                f.write(single_line(dataset))
    else:
        with open(filename,'w') as f:
            for dataset in data_list:
                f.write(single_line(dataset))

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

    return xcoords*np.cos(angle)+ycoords*np.sin(angle),-xcoords*np.sin(angle)+ycoords*np.cos(angle)

def dr(x1,x2,y1,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def img_sept(x,y):

    return np.sort(np.array([dr(x[0],x[1],y[0],y[1]),dr(x[0],x[2],y[0],y[2]),dr(x[0],x[3],y[0],y[3]),
         dr(x[1],x[2],y[1],y[2]),dr(x[1],x[3],y[1],y[3]),dr(x[2],x[3],y[2],y[3])]))

def identify(x,y,RE):

    separations = img_sept(x,y)

    if separations[0] > RE:
        return 0

    if separations[1] <= 1.15*RE:
        return 2

    elif separations[0] <= 0.85*RE:
        return 1

    else:
        return 0

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
            #print(row,fname)
            #row_split = filter(None, row)
            row_split = list(filter(None, row))
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

def filter_by_position(lens_components, x_filter=None, y_filter=None, mindis_front=0.5, mindis_back=0.3, log_masscut_low=7,
                       zmain=None, cosmology=None):
    """
    :param xsub: sub x coords
    :param ysub: sub y coords
    :param x_filter: img x coords
    :param y_filter: img y coords
    :param mindis: max 2d distance
    :return: filtered subhalos
    """

    masscut_low = 10**log_masscut_low

    keep_index = []

    for index, deflector in enumerate(lens_components):

        if not deflector.is_subhalo:
            keep_index.append(index)
            continue

        if zmain >= deflector.redshift:

            """
            for LOS halos; keep if it's rescaled position is near an image
            """

            scale = np.ones_like(x_filter)
            _mindis = mindis_front

        else:
            #zmain < deflector.redshift:

            """
            for halos behind the main lens
            """

            beta = cosmology.beta(deflector.redshift,zmain,cosmology.zsrc)

            scale = np.ones_like(x_filter)*(1 - beta)
            _mindis = mindis_back
            #scale_mindis = 0.5


        x, y = deflector.lenstronomy_args['center_x'], deflector.lenstronomy_args['center_y']

        for i in range(0, len(x_filter)):

            dr = ((x - x_filter[i]*scale[i]) ** 2 + (y - y_filter[i]*scale[i]) ** 2) ** .5

            if dr <= _mindis or deflector.other_args['mass'] >= masscut_low:
                keep_index.append(index)

                break

    newcomponents = [lens_components[i] for i in keep_index]
    new_redshift_list = [lens_components[i].redshift for i in keep_index]

    return newcomponents, new_redshift_list

def copy_directory(dirname,location):

    shutil.copy(dirname,location)

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def delete_dir(dirname=''):

    shutil.rmtree(dirname)

def rebin_image(image,factor):

    if np.shape(image)[0]%factor != 0:
        raise ValueError('size of image must be divisible by factor')

    def rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]

        return a.reshape(sh).mean(-1).mean(1)
    size = int(np.shape(image)[0]*factor**-1)

    return rebin(image,[size,size])

def convolve_image(image,kernel='Gaussian',scale=None):

    if kernel == 'Gaussian':
        grid = sfilt.gaussian_filter(image, scale * (2.355) ** -1, mode='constant', cval=0)
    elif kernel == 'HST':
        grid = sfilt.gaussian_filter(image, scale * (2.355) ** -1, mode='constant', cval=0)

    return grid

def nfw_kr(X):
    def f(x):

        if isinstance(x, int) or isinstance(x, float):
            if x > 1:
                return np.arctan((x ** 2 - 1) ** .5) * (x ** 2 - 1) ** -.5
            elif x < 1:
                return np.arctanh((1 - x ** 2) ** .5) * (1 - x ** 2) ** -.5
            else:
                return 1
        else:
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)

            vals = np.ones_like(x)
            flow = (1 - x[inds1] ** 2) ** .5
            fhigh = (x[inds2] ** 2 - 1) ** .5

            vals[inds1] = np.arctanh(flow) * flow ** -1
            vals[inds2] = np.arctan(fhigh) * fhigh ** -1

            return vals

    return 2 * (1 - f(X)) * (X ** 2 - 1) ** -1

def ellipse_coordinates(coordinatesx,coordinatesy,rmax,q=0.08,theta=None):

    a = rmax
    b = q*rmax
    x,y = rotate(coordinatesx,coordinatesy,theta)
    ellipse = (x*a**-1) ** 2 + (y*b**-1) ** 2 <= 1
    #return x[ellipse],y[ellipse]
    return ellipse

def phi_q2_ellipticity(phi, q):
    """

    :param phi:
    :param q:
    :return:
    """
    e1 = (1.-q)/(1.+q)*np.cos(2*phi)
    e2 = (1.-q)/(1.+q)*np.sin(2*phi)
    return e1, e2


def ellipticity2phi_q(e1, e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    if c > 0.999:
        c = 0.999
    q = (1-c)/(1+c)
    return phi, q

def find_closest_xy(ximg,yimg,xref,yref):

    dx = np.array(ximg) - xref
    dy = np.array(yimg) - yref

    dr = np.sqrt(dx**2+dy**2)

    return np.argsort(dr)

def chi_square_img(data1x,data1y,data2x,data2y,sigma,reorder=False):

    if len(data1x) != len(data2x):
        return 10**10

    if reorder:
        inds = sort_image_index(data1x,data1y,data2x,data2y)
        data1x = data1x[inds]
        data1y = data1y[inds]

    return np.sum(((data1x - data2x)*sigma**-1)**2 + ((data1y - data2y)*sigma**-1)**2)

def chi_square(d1,d2,sigma):

    return np.sum(((d1 - d2)*sigma**-1)**2)

def flux_at_edge(image):

    maxbright = np.max(image)
    edgebright = [image[0,:],image[-1,:],image[:,0],image[:,-1]]

    for edge in edgebright:
        if any(edge > maxbright * 0.333):
            return True
    else:
        return False






