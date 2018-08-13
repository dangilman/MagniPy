from MagniPy.Analysis.PresetOperations.fluxratio_distributions import compute_fluxratio_distributions
from MagniPy.paths import *
import sys
from time import time
import numpy as np

Ntotal = 2
t0 = time()

grid_rmax = 0.12
grid_res = 0.001

zlens,zsrc = 0.5,1.5

filter_halos = False
start_shear = 0.08
mindis = 0.5
log_masscut_low = 7

x_image = np.array([-0.49598, 0.14184, -1.05127, 0.90563])
y_image = np.array([ 0.97263, 1.04017, 0.11488, -1.22083])
mag = np.array([1, 0.676482, 0.376914, 0.198391])
datatofit = [x_image,y_image,mag,np.array([0,0,0,0])]

massprofile = 'TNFW'
identifier = 'filtered'

halo_models = ['plaw_main','plaw_main','plaw_main','plaw_main','composite_plaw','composite_plaw']
normalizations = [0,0.01,0.02,0.01,0.01,0.01]
outfilenames = ['null_test','CDM_main_tester','CDMx2_main','WDM_main','CDM_los','WDM_los']
logmhm_values = [0,0,0,8,0,8]

for (mod,fsub,outfilename,logmhm) in zip(halo_models,normalizations,outfilenames,logmhm_values):

    model_args = {'fsub':fsub,'log_mL':7,'log_mH':10,'r_core':'0.5Rs','logmhm':logmhm}

    compute_fluxratio_distributions(massprofile=massprofile,halo_model=mod,model_args=model_args,
                                data2fit=datatofit,Ntotal=Ntotal,outfilename=outfilename,zlens=zlens,zsrc=zsrc,
                                identifier=identifier,grid_rmax=grid_rmax,res=grid_res,source_size=0.0012*2.3**-1,
                                filter_halo_positions=filter_halos,outfilepath=fluxratio_data_path,write_to_file=True,
                                mindis=mindis,log_masscut_low=log_masscut_low,start_shear=start_shear)




