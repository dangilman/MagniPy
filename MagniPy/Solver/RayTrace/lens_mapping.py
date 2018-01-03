import lens_profiles
from fits_handle import *
from lens_simulations.data_visualization.data_handling import *

class Gridspecs:

    def __init__(self,gridres=False,gridsize=False,macro_gridsize=False,default_res = 0.0004,default_size=1200):

        if macro_gridsize is False:
            self.macro_gridsize=default_size
        else:
            self.macro_gridsize = macro_gridsize
        if gridres is False:
            self.gridres = default_res
        else:
            self.gridres = gridres
        if gridsize is False:
            self.gridsize = default_size
        else:
            self.gridsize = gridsize

class Deflector:

    def __init__(self, gridsize=False, gridres=False, **kwargs):

        # sets up a deflector class, specifies grid and macromodel
        gridres = kwargs['gridres'] if 'gridres' in kwargs else gridres
        gridsize = kwargs['gridsize'] if 'gridsize' in kwargs else gridsize
        macrogridsize = kwargs['macro_gridsize'] if 'macro_gridsize' in kwargs else 1200
        gridspecs = Gridspecs(gridsize=gridsize,gridres=gridres,macro_gridsize=macrogridsize)

        self.gridres = gridspecs.gridres
        self.gridsize = gridspecs.gridsize
        self.macro_gridsize = gridspecs.macro_gridsize


        if 'at_image_only' in kwargs:

            self.global_macro = False
            self.macro_profile = kwargs['macro_profile']

        else:

            self.global_macro = True

            if 'macro_profile' in kwargs:

                macrodef = DefMap(ximg=0, yimg=0, grid_res=self.gridres, gridsize=gridspecs.macro_gridsize)
                macrodef.set_macromodel_profile(kwargs['macro_profile'])

                macro_profile=kwargs['macro_profile']

                self.macro_profile = kwargs['macro_profile']
                self.macrox,self.macroy = macrodef.update_defxy(profile=macro_profile, macromodel=True, x=0,y=0, passback=True, **kwargs)

            else:

                self.xdef,self.gridres = open_fits(kwargs['path_to_defmapx'])
                self.ydef, res = open_fits(kwargs['path_to_defmapy'])

                self.gridsize = gridspecs.gridsize
                self.macro_gridsize = int(np.shape(self.xdef)[0]*.5*self.gridres*1000)

                #if self.gridres!=gridres:
                #    print 'setting gridres equal: ',self.gridres

                self.macrox = self.xdef
                self.macroy = self.ydef


    def at_image_location(self, xref, yref, macroparams = {}, **kwargs):
        # xfit, yfit, b=b, x_center=x0, y_center=y0,
                                   #ellip=e, ellip_theta=ePA, shear=s, shear_theta=sPA, rcore=rc, rtrunc=rt
        # sets the deflection map at each point to the macrox, macroy values

        self.full_lensmap=False

        xref,yref = np.squeeze(xref),np.squeeze(yref)

        self.basemap1 = DefMap(ximg=xref[0], yimg=yref[0], grid_res=self.gridres, gridsize=self.gridsize)
        self.basemap2 = DefMap(ximg=xref[1], yimg=yref[1], grid_res=self.gridres, gridsize=self.gridsize)
        self.basemap3 = DefMap(ximg=xref[2], yimg=yref[2], grid_res=self.gridres, gridsize=self.gridsize)
        self.basemap4 = DefMap(ximg=xref[3], yimg=yref[3], grid_res=self.gridres, gridsize=self.gridsize)

        if self.global_macro:

            self.basemap1.update_defxy(precomputed=True, xdef=self.macrox, ydef=self.macroy, asec_perpixel=self.gridres,
                                       x=xref[0], y=yref[0], readin=True)

            self.basemap2.update_defxy(precomputed=True, xdef=self.macrox, ydef=self.macroy, asec_perpixel=self.gridres,
                                       x=xref[1], y=yref[1], readin=True)

            self.basemap3.update_defxy(precomputed=True, xdef=self.macrox, ydef=self.macroy, asec_perpixel=self.gridres,
                                       x=xref[2], y=yref[2], readin=True)

            self.basemap4.update_defxy(precomputed=True, xdef=self.macrox, ydef=self.macroy, asec_perpixel=self.gridres,
                                       x=xref[3], y=yref[3], readin=True)

        else:

            self.basemap1.set_macromodel_profile(self.macro_profile)
            self.basemap2.set_macromodel_profile(self.macro_profile)
            self.basemap3.set_macromodel_profile(self.macro_profile)
            self.basemap4.set_macromodel_profile(self.macro_profile)

            self.basemap1.update_defxy(macromodel=True,lensparams=macroparams)

            self.basemap2.update_defxy(macromodel=True,lensparams=macroparams)

            self.basemap3.update_defxy(macromodel=True,lensparams=macroparams)

            self.basemap4.update_defxy(macromodel=True,lensparams=macroparams)

        self.defmapx = np.zeros((np.shape(self.basemap1.defmapx)[0], np.shape(self.basemap1.defmapx)[1], 4))
        self.defmapy = np.zeros((np.shape(self.basemap1.defmapy)[0], np.shape(self.basemap1.defmapy)[1], 4))
        self.xcoords = np.zeros((np.shape(self.basemap1.defmapx)[0], np.shape(self.basemap1.defmapx)[1], 4))
        self.ycoords = np.zeros((np.shape(self.basemap1.defmapy)[0], np.shape(self.basemap1.defmapy)[1], 4))

        self.xcoords[:, :, 0], self.xcoords[:, :, 1], self.xcoords[:, :, 2], self.xcoords[:, :,
                                                                             3] = self.basemap1.xcoords,self.basemap2.xcoords,self.basemap3.xcoords,self.basemap4.xcoords
        self.ycoords[:, :, 0], self.ycoords[:, :, 1], self.ycoords[:, :, 2], self.ycoords[:, :,
                                                                             3] = self.basemap1.ycoords,self.basemap2.ycoords,self.basemap3.ycoords,self.basemap4.ycoords

        self.reset_defmap()

    def reset_defmap(self):

        if self.full_lensmap:

            self.defmapx,self.defmapy = self.self.macrox,self.self.macroy

        else:

            self.defmapx[:, :, 0], self.defmapx[:, :, 1], self.defmapx[:, :, 2], self.defmapx[:, :,
                                                                                     3] = self.basemap1.defmapx, self.basemap2.defmapx, self.basemap3.defmapx, self.basemap4.defmapx
            self.defmapy[:, :, 0], self.defmapy[:, :, 1], self.defmapy[:, :, 2], self.defmapy[:, :,
                                                                                     3] = self.basemap1.defmapy, self.basemap2.defmapy, self.basemap3.defmapy, self.basemap4.defmapy

    def map_full_lens(self):

        self.full_lensmap=True

        print 'laying down coordinates with dimension: '+str((self.macro_gridsize))+' by '+str((self.macro_gridsize))
        self.basemap = DefMap(ximg=0, yimg=0, grid_res=self.gridres, gridsize=self.macro_gridsize)

        self.xcoords,self.ycoords = self.basemap.xcoords,self.basemap.ycoords
        self.defmapx = self.macrox
        self.defmapy = self.macroy

    def add_subhalos(self, subx, suby, sub_b, subtrunc, subcore, subprofile=None):

        if subprofile is None:
            raise ValueError('must specify a subhalo mass profile')

        subhalo_args = lens_profiles.lensmod_to_kwargs(subx, suby, sub_b, subtrunc, subcore, subprofile=subprofile)

        self.reset_defmap()

        if self.full_lensmap:
            for i in range(0,len(subx)):
                xadd,yadd = self.basemap.update_defxy(subhalo_args, passback=True)
                self.defmapx+=xadd
                self.defmapy+=yadd

        else:

            if self.basemap1.subhalotype_set is False:
                self.basemap1.set_subhalo_profile(subprofile)
            if self.basemap2.subhalotype_set is False:
                self.basemap2.set_subhalo_profile(subprofile)
            if self.basemap3.subhalotype_set is False:
                self.basemap3.set_subhalo_profile(subprofile)
            if self.basemap4.subhalotype_set is False:
                self.basemap4.set_subhalo_profile(subprofile)

            for i in range(0, len(subx)):


                single_arg = {}

                for pname,values in subhalo_args.items():
                    single_arg.update({pname:values[i]})

                xadd, yadd = self.basemap1.update_defxy(precomputed=False, passback=True, lensparams=single_arg)

                self.defmapx[:, :, 0] += xadd
                self.defmapy[:, :, 0] += yadd

                xadd, yadd = self.basemap2.update_defxy(precomputed=False, passback=True, lensparams=single_arg)
                self.defmapx[:, :, 1] += xadd
                self.defmapy[:, :, 1] += yadd

                xadd, yadd = self.basemap3.update_defxy(precomputed=False, passback=True, lensparams=single_arg)
                self.defmapx[:, :, 2] += xadd
                self.defmapy[:, :, 2] += yadd
                xadd, yadd = self.basemap4.update_defxy(precomputed=False, passback=True, lensparams=single_arg)
                self.defmapx[:, :, 3] += xadd
                self.defmapy[:, :, 3] += yadd

    def img_mag(self, src_mod='gaussian', srcx=float, srcy=float, src_size=False):

        width = src_size

        mag1 = compute_mag(xcoords=self.xcoords[:,:,0], ycoords=self.ycoords[:,:,0], defx=self.defmapx[:, :, 0],
                           defy=self.defmapy[:, :, 0], source_mod=src_mod, width=width, srcx=srcx, srcy=srcy,
                           refgrid=self.basemap1.refgrid)
        mag2 = compute_mag(xcoords=self.xcoords[:,:,1], ycoords=self.ycoords[:,:,1], defx=self.defmapx[:, :, 1],
                           defy=self.defmapy[:, :, 1], source_mod=src_mod, width=width, srcx=srcx, srcy=srcy,
                           refgrid=self.basemap1.refgrid)
        mag3 = compute_mag(xcoords=self.xcoords[:,:,2], ycoords=self.ycoords[:,:,2], defx=self.defmapx[:, :, 2],
                           defy=self.defmapy[:, :, 2], source_mod=src_mod, width=width, srcx=srcx, srcy=srcy,
                           refgrid=self.basemap1.refgrid)
        mag4 = compute_mag(xcoords=self.xcoords[:,:,3], ycoords=self.ycoords[:,:,3], defx=self.defmapx[:, :, 3],
                           defy=self.defmapy[:, :, 3], source_mod=src_mod, width=width, srcx=srcx, srcy=srcy,
                           refgrid=self.basemap1.refgrid)

        return np.array([mag1, mag2, mag3, mag4])

    def show_full_lens(self,srcmod='gaussian',srcsize=False,srcx=float,srcy=float):
        if srcsize is False:
            srcsize=self.srcsize
        print 'mapping to image plane...'

        image = map_to_imgplane(self.xcoords,self.ycoords,self.defmapx,self.defmapy,source_mod=srcmod,width=srcsize,srcx=srcx,srcy=srcy)

        return image


class DefMap:

    def __init__(self,ximg=float,yimg=float,grid_res=float,gridsize=int):
        # specify a grid location (xmin,xmax,ymin,ymax) and a grid spacing;
        # each point is associated with a deflection angle
        # grid res: arcsec per pixel
        # gridsize: m.a.s.

        gridsize*=.001
        self.gridsize=gridsize
        steps = max(1,2*round(gridsize*grid_res**-1))
        self.xcoords, self.ycoords = np.meshgrid(np.linspace(ximg-gridsize,ximg+gridsize,steps),np.linspace(yimg-gridsize,yimg+gridsize,steps))
        xref,yref = np.meshgrid(np.linspace(-gridsize,gridsize,steps),-np.linspace(-gridsize,gridsize,steps))
        self.refgrid=np.sqrt(xref**2+yref**2)
        self.defmapx,self.defmapy = np.zeros_like(self.xcoords),np.zeros_like(self.ycoords)
        self.steps=steps
        self.gridres=grid_res
        self.subhalotype_set=False
        self.macrotype_set = False

    def set_subhalo_profile(self,profile=None):
        self.subhalotype_set = True

        if profile == 'pjaffe':
            self.clump_defmap = lens_profiles.pjaffe_profile(self.xcoords, self.ycoords)
        elif profile == 'SIE':
            self.clump_defmap = lens_profiles.SIE_profile(self.xcoords, self.ycoords)
        elif profile == 'simonSIE':
            self.clump_defmap = lens_profiles.simonSIE_profile(self.xcoords, self.ycoords)
        elif profile == 'nfw':
            self.clump_defmap = lens_profiles.NFW(self.xcoords, self.ycoords)
        elif profile == 'ptmass':
            self.clump_defmap = lens_profiles.PointMass(self.xcoords, self.ycoords)
        elif profile=='tnfw' or profile=='tnfw3':
            self.clump_defmap = lens_profiles.tNFW(self.xcoords,self.ycoords)
        else:
            raise ValueError('profile '+profile+' not recognized')
    def set_macromodel_profile(self,profile):
        self.macrotype_set = True
        if profile == 'pjaffe':
            self.clump_defmap_macro = lens_profiles.pjaffe_profile(self.xcoords, self.ycoords)
        elif profile == 'SIE':
            self.clump_defmap_macro = lens_profiles.SIE_profile(self.xcoords, self.ycoords)
        elif profile == 'simonSIE':
            self.clump_defmap_macro = lens_profiles.simonSIE_profile(self.xcoords, self.ycoords)
        elif profile == 'nfw':
            self.clump_defmap_macro = lens_profiles.NFW(self.xcoords, self.ycoords)
        elif profile == 'ptmass':
            self.clump_defmap = lens_profiles.PointMass(self.xcoords, self.ycoords)

    def update_defxy(self, precomputed=False, passback=False, xdef=None, ydef=None, asec_perpixel=None, x=None, y=None,
                     readin=False, macromodel=False, lensparams={}):

        if precomputed:
            if readin:

                xdeffull,ydeffull,scale,x0,y0 = xdef,ydef,asec_perpixel,x,y
                shift=len(xdeffull)*.5
                x0,y0=round(x0*self.gridres**-1+shift),round(y0*self.gridres**-1+shift)

                xdef = xdeffull[int(y0-self.steps*.5):int(y0+self.steps*.5),int(x0-self.steps*.5):int(x0+self.steps*.5)]

                ydef = ydeffull[int(y0-self.steps*.5):int(y0+self.steps*.5),int(x0-self.steps*.5):int(x0+self.steps*.5)]

        else:

            if macromodel:
                xdef,ydef = self.clump_defmap_macro.def_angle(**lensparams)

            else:

                xdef, ydef = self.clump_defmap.def_angle(**lensparams)

        if passback:
            return xdef, ydef
        else:

            self.defmapx = self.defmapx+xdef
            self.defmapy = self.defmapy+ydef


def map_to_source(xgrid,ygrid,defmapx,defmapy):
    # xgrid and ygrid: x coordinates and ycoordinates in image plane
    # defmapxy = deflection map
    if np.shape(xgrid)!=np.shape(defmapx):
        raise ValueError('x grids must be same size')
    elif np.shape(ygrid)!=np.shape(defmapy):
        raise ValueError('y grids must be same size')

    return xgrid-defmapx,ygrid-defmapy


def normal_dis(x=0, y=0, sigma=1, xmean=0, ymean=0, r=False):
    if r is False:
        return (2 * np.pi * sigma ** 2) ** -1 * np.exp(
            -.5 * (x - xmean) ** 2 * sigma ** -2 - .5 * (y - ymean) ** 2 * sigma ** -2)
    else:
        return (2 * np.pi * sigma ** 2) ** -1 * np.exp(-.5 * (r * sigma ** -1) ** 2)

def sum_srcflux(xgrid,ygrid,**kwargs):
    # input points in the source plane xgrid ygrid, sum the flux

    if kwargs['source_mod']=='gaussian':

        x0, y0 = kwargs['srcx'], kwargs['srcy']
        size = kwargs['width']*2.355**-1
        #print kwargs['width']

        dx, dy = xgrid - x0, ygrid - y0

        #import matplotlib.pyplot as plt
        #x_sr = size*.5*np.cos(np.linspace(0,2*np.pi,100))
        #y_sr = size*.5* np.sin(np.linspace(0, 2 * np.pi, 100))
        #plt.scatter(dx,dy)
        #plt.scatter(x_sr,y_sr,color='r')
        #plt.show()

        flux = np.sum(normal_dis(x=dx,y=dy,sigma=size))
        #import matplotlib.pyplot as plt
        #plt.imshow(normal_dis(x=dx,y=dy,sigma=size))
        #plt.show()
        norm = np.sum(normal_dis(sigma=size,r=kwargs['refgrid']))

        return flux*norm**-1

    elif kwargs['source_mod']=='disk':
        x0, y0 = kwargs['srcx'], kwargs['srcy']
        radius = .5*kwargs['width']
        dx, dy = xgrid - x0, ygrid - y0
        norm=np.sum(kwargs['refgrid']<radius)
        r = (dx**2+dy**2)**.5

        return np.sum(r < radius)*norm**-1

def asec_to_pixel(scale,shift,x,y):
    return x*scale**-1+shift,y*scale**-1+shift

def compute_mag(xcoords,ycoords,defx,defy,**kwargs):
    # input (x,y) points, (srcx,srcy),
    # input the xy grid, and deflection grids, compute magnifications

    srcx_coords,srcy_coords = map_to_source(xcoords,ycoords,defx,defy)

    return sum_srcflux(srcx_coords,srcy_coords,**kwargs)

def map_to_imgplane(xgrid,ygrid,defx,defy,**kwargs):
    """

    :param xgrid: the x coordinates in the image plane; for individual images, could be several small grids, or one big one
    :param ygrid: see above
    :param defx: deflection angles in the image plane in the specified xgrid
    :param defy: see above
    :param kwargs: source_mod (gaussian,light); if light: kwargs['source_light']=array_like xgrid
    :return:
    """
    if kwargs['source_mod']=='gaussian':
        x0, y0 = kwargs['srcx'],kwargs['srcy']
        size = kwargs['width'] * 2.355 ** -1
        dx, dy = xgrid - defx - x0, ygrid -defy - y0

        image = normal_dis(x=dx, y=dy, sigma=size)

    elif kwargs['source_mod']=='sersic':
        x0, y0 = kwargs['srcx'], kwargs['srcy']
        re = kwargs['width']
        dx, dy = xgrid - defx - x0, ygrid - defy - y0
        def bn(n):
            return 1.9992 * n - 0.3271 + 4 * (405 * n) ** -1
        def sers(x,re=float,n=4):
            return np.exp(bn(n)*((x*re**-1)**(1*n**-1)-1))
        r = np.sqrt(dx**2+dy**2)
        return sers(r,re)

    elif kwargs['source_mod']=='light':
        from scipy import interpolate
        x0, y0 = kwargs['srcx'], kwargs['srcy']
        dx,dy = xgrid-defx-x0,ygrid-defy-y0
        # now evaluate dx,dy at points in source plane, points interpolated from source grid map
        pix_scale_src = kwargs['pixscale_source']
        pix_scale_img = kwargs['pixscale_img']
        shrink = pix_scale_img*pix_scale_src**-1
        x,y = np.linspace(-int(kwargs['width']*.5),int(kwargs['width']*.5),int(np.shape(kwargs['source_light'])[0]))
        xx,yy = np.meshgrid(x,y)
        f = interpolate.interp2d(xx,yy,kwargs['source_light'], kind='cubic')
        image = f(dx,dy)


    return image

def filter_subs(ximg,yimg,subx,suby,mindis=.5):

    inds = []

    for s in range(0,np.shape(subx)[0]):

        if any(np.sqrt((ximg-subx[s])**2 + (yimg-suby[s])**2)<mindis):
            inds.append(s)

    if len(inds)==0:
        return False
    else:
        return inds


def get_SIEmags(xfit, yfit, subhalos, SIEfit, srcfit, fluxerr=0, src_size=0.0012,subprofile='',shrcoords=1,gridsize=40,
                print_status=True):

    if src_size<0.0014:
        gridsize=40
    else:
        gridsize=100

    if isinstance(subhalos,list):
        pass
    else:
        subhalos=[subhalos]

    if np.array(xfit).ndim == 1:
        mag_final = np.zeros((1, 4))
    else:
        mag_final = np.zeros((np.shape(xfit)[0], 4))

    for n, subs in enumerate(subhalos):
        if print_status:
            print 'computing mag #: ', n

        subs = np.squeeze(subs)

        b, x0, y0, e, ePA, s, sPA, rc, rt = parse_macro(SIEfit[n],shrcoords)

        lens = Deflector(macro_profile='SIE', gridsize=gridsize, gridres=0.0004, at_image_only=True)
        macro_args = lens_profiles.lensmod_to_kwargs(x0, y0, b, rt, rc,ellip=e,ellip_PA=ePA,shear=s,shear_theta=sPA,subprofile='SIE')

        if np.array(xfit).ndim == 1:
            lens.at_image_location(xfit, yfit, macroparams=macro_args)
        else:
            lens.at_image_location(xfit[n, :], yfit[n, :], macroparams=macro_args)

        if len(subs) > 0:


            if subs.ndim == 1:

                lens.add_subhalos(subx=[subs[4]], suby=[subs[5]], sub_b=[subs[3]], subtrunc=[subs[1]],
                                  subcore=[subs[2]],subprofile=subprofile)
            else:

                lens.add_subhalos(subx=subs[:, 4], suby=subs[:, 5], sub_b=subs[:, 3], subtrunc=subs[:, 1],
                                      subcore=subs[:, 2],subprofile=subprofile)

        if np.array(xfit).ndim == 1:

            mag_final = lens.img_mag(src_size=src_size, srcx=srcfit[0],
                                     srcy=srcfit[1])

            mag_final *= max(mag_final) ** -1

            if fluxerr!=0:
                for i in range(0,len(mag_final)):
                    mag_final[i] += np.random.normal(0,fluxerr*mag_final[i])

        else:
            mag_final[n, :] = lens.img_mag(src_size=src_size, srcx=srcfit[n, 0],
                                           srcy=srcfit[n, 1])

            mag_final[n, :] *= max(mag_final[n, :]) ** -1

            if fluxerr!=0:

                for i in range(0,len(mag_final[n,:])):

                    mag_final[n,i] += np.random.normal(0,fluxerr*mag_final[n,i])

                mag_final[n,:]*=max(mag_final[n,:])**-1

    return mag_final





