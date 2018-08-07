from MagniPy.Analysis.PresetOperations.fluxratio_distributions import compute_fluxratio_distributions
from MagniPy.paths import *
import sys
from time import time
import numpy as np

Ntotal = 500
t0 = time()

grid_rmax = 0.12
grid_res = 0.001

zlens,zsrc = 0.5,1.5

filter = True
start_shear = 0.08
mindis = 0.5
log_masscut_low = 7

datatofit = [[0.4043, 0.96909, -0.43733, -0.30671],[1.0384, 0.51127, 1.00532, -0.85602],
             [1.0, 0.724108, 0.670567, 0.183723],
             [0, 0, 0, 0]]
sigmas = [[0.003]*4]*2,[0.3]*4,[0.1,100,100,100]

Ntotal = 500
t0 = time()

grid_rmax = 0.12
grid_res = 0.001

zlens,zsrc = 0.5,1.5

filter = True
start_shear = 0.08
mindis = 0.5
log_masscut_low = 7

datatofit = [[0.4043, 0.96909, -0.43733, -0.30671],[1.0384, 0.51127, 1.00532, -0.85602],
             [1.0, 0.724108, 0.670567, 0.183723],
             [0, 0, 0, 0]]
sigmas = [[0.003]*4]*2,[0.3]*4,[0.1,100,100,100]

massprofile = 'TNFW'
identifier = 'LOS'

outfilename = 'CDM_test'
halo_model = 'composite_plaw'
model_args = {'fsub':0.01,'log_mL':7,'log_mH':10,'r_core':'0.5Rs','logmhm':0}

compute_fluxratio_distributions(massprofile=massprofile,halo_model=halo_model,model_args=model_args,
                                data2fit=datatofit,Ntotal=Ntotal,outfilename=outfilename,zlens=zlens,zsrc=zsrc,
                                identifier=identifier,grid_rmax=grid_rmax,res=grid_res,source_size=0.0012*2.3**-1,
                                filter_halo_positions=filter,outfilepath=fluxratio_data_path,write_to_file=True,
                                mindis=mindis,log_masscut_low=log_masscut_low,start_shear=start_shear)





