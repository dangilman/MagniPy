import numpy as np
from MagniPy.MassModels.PointMass import PTmass
from MagniPy.LensBuild.Cosmology.cosmology import Cosmo
from MagniPy.LensBuild.lens_assemble import Deflector

def number_per_image(f_pbh=float,redshift=None,zsrc=None,cosmology_class=classmethod, M=float, R_ein=None, distance_factor=25):

    k_pbh = f_pbh*0.5

    if R_ein is None:
        ptmass = PTmass(z = redshift, zsrc = zsrc)
        R_ein = ptmass.R_ein(M)

    return np.pi*k_pbh*cosmology_class.get_sigmacrit_z1z2(redshift,zsrc)*((distance_factor*R_ein)**2)*M**-1,distance_factor*R_ein

def drawPBH(ximgs=[],yimgs=[],f_pbh=float,redshift=None,zsrc=None,cosmology=classmethod,mass=float,distance_factor=25,Nrealizations=1):

    ptmass = PTmass(z=redshift, zsrc=zsrc)

    R_ein = ptmass.R_ein(mass)

    N = number_per_image(f_pbh=f_pbh, cosmology_class=cosmology, M=mass,R_ein=R_ein,distance_factor=distance_factor)

    realizations = []

    for r in range(0,Nrealizations):

        black_holes = []

        for imgnum in range(0,len(ximgs)):

            ximg,yimg = ximgs[imgnum],yimgs[imgnum]

            draw = True

            n = N

            while draw:

                if n >= 1:
                    draw = True
                    n = n-1
                elif n<1 and n>0:
                    if n >= np.random.rand():
                        draw = True
                        n = n-1
                    else:
                        draw = False
                else:
                    draw = False

                if draw:
                    R_ein = ptmass.R_ein(mass)
                    x = np.random.uniform(ximg-distance_factor*R_ein,ximg+distance_factor*R_ein)
                    y = np.random.uniform(yimg - distance_factor * R_ein, yimg + distance_factor * R_ein)

                    lens_kwargs = {'x':x,'y':y,'mass':mass}

                    black_holes.append(Deflector(subclass = PTmass(), redshift=cosmology.zd, is_subhalo=True, **lens_kwargs))

        realizations.append(black_holes)

    return realizations
