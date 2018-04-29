from MagniPy.LensBuild.defaults import *
from MagniPy.LensBuild.Cosmology.halo_mass_function import *
from PBHgen import *

def mainlens_plaw(fsub,plaw_index,cosmo,kappa_Rein=0.5, log_mL = None, log_mH = None):

    mH,mL = 10**log_mH,10**log_mL

    A0_perasec2 = fsub*kappa_Rein*cosmo.sigmacrit*(2+plaw_index)*(mH**(2+plaw_index) - mL**(2+plaw_index))**-1

    return A0_perasec2,plaw_index

def LOS_plaw(zmin,zmax,zmain,zsrc,rescale_sigma8=False,omega_M_void=None,log_mL=None,log_mH=None,cone_base=None):

    if zmax >= zsrc:
        zmax = zsrc - 1e-3

    if zmin == 0:
        zmin = 1e-3
    zstep = 0.02
    zvals = np.linspace(zmin,zmax,int(np.ceil((zmax-zmin)*zstep**-1)+1))

    HMF = HaloMassFunction(sigma_8=default_sigma8, zd=zmain, zsrc=zsrc,
                           rescale_sigma8=rescale_sigma8, omega_M_void=omega_M_void)

    M = np.logspace(log_mL,log_mH,30)

    zstep = zvals[1]-zvals[0]

    A0_z = []
    plaw_index_z = []

    for z in zvals:

        dn_dm = HMF.dndM_integrated_z1z2(M,z,z+zstep,cone_base=cone_base)

        _,normalization,index = HMF.mass_function_moment(M,dn_dm,0)

        A0_z.append(normalization)
        plaw_index_z.append(index)

    return A0_z,plaw_index_z,zvals

def delta_main(M,f_pbh,ximgs,yimgs,omega,zmin,zmax,zmain,zsrc,cosmo):

    return drawPBH(ximgs=ximgs,yimgs=yimgs,f_pbh=f_pbh,cosmology=cosmo,mass=M)

def LOS_delta(M,omega,zmin,zmax,zmain,zsrc,cone_base):

    from MagniPy.LensBuild.defaults import zstep

    if zmax>= zsrc:
        zmax = zsrc-1e-3

    if zmin==0:
        zmin = 1e-3

    zvals = np.linspace(zmin,zmax,int(np.ceil((zmax-zmin)*zstep**-1)+1))

    HMF = HaloMassFunction(sigma_8=default_sigma8, zd=zmain, zsrc=zsrc, )

    zstep = zvals[1]-zvals[0]

    N_z = []

    for z in zvals:

        Nz = HMF.dndM_integrated_z1z2(M,z,z+zstep,functype='delta',omega=omega,cone_base=cone_base)

        N_z.append(Nz)

    return N_z,zvals


