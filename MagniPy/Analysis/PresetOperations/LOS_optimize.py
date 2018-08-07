# coding: utf-8

# In[1]:

from MagniPy.Analysis.PresetOperations.halo_constructor import Constructor
from MagniPy.LensBuild.defaults import *
from MagniPy.Solver.analysis import Analysis
from MagniPy.Solver.solveroutines import SolveRoutines
from MagniPy.util import write_fluxes
from MagniPy.paths import *
import matplotlib.pyplot as plt

# In[ ]:

zlens, zsrc = 0.5, 1.5
analysis = Analysis(zlens, zsrc)
solver = SolveRoutines(zlens, zsrc)
multiplane = True

realization = Constructor(zlens, zsrc, LOS_mass_sheet=True)
macroargs_start = {'R_ein': 1.2, 'x': 0, 'y': 0, 'ellip': 0.22, 'ellip_theta': 23, 'shear': 0.06, 'shear_theta': -40,
                   'gamma': 2}
macromodel_start = Deflector(redshift=0.5, subclass=SIE(), varyflags=['1', '1', '1', '1', '1', '1', '1', '0', '0', '0'],
                             **macroargs_start)

N = 10
srcx,srcy = 0.12,-0.175

datatofit = solver.solve_lens_equation(macromodel=macromodel_start,realizations=None,multiplane=False,srcx=srcx,srcy=srcy)

arg_list = []
fsubvals = [0.01]
logmhmvals = [0]
outfilenames = ['CDM_run_withmainhalos']
model_types = ['composite_plaw']

assert len(outfilenames) == len(logmhmvals)
assert len(fsubvals) == len(logmhmvals)
assert len(model_types) == len(fsubvals)
assert datatofit[0].nimg == 4

for i in range(0,len(outfilenames)):

    flux_anomaly = []
    outfilename = outfilenames[i]
    fsub = fsubvals[i]
    logmhm = logmhmvals[i]
    model_type = model_types[i]
    shears,shear_pa,xcen,ycen = [],[],[],[]

    while len(flux_anomaly)<N:

        halos = realization.render('TNFW', 'composite_plaw', {'fsub': fsub, 'r_core': '0.5Rs', 'logmhm': logmhm},
                                   Nrealizations=1)

        datatofit = solver.solve_lens_equation(macromodel=macromodel_start, realizations=halos,
                                               multiplane=multiplane, srcx=srcx, srcy=srcy)

        if datatofit[0].nimg == 4:


            optdata, optmodel = solver.two_step_optimize(datatofit=datatofit[0], realizations=None,macromodel=get_default_SIE(0.5),
                                                         multiplane=False, method='lensmodel',grid_rmax=0.1)
            #shear1 = 0.5 * (fxx - fyy)
            #shear2 = -0.5 * fxy
            #shear_LOS.append(np.sqrt(shear1 ** 2 + shear2 ** 2))
            #shear_fit.append(optmodel[0].lens_components[0].shear)
            flux_anomaly.append(optdata[0].flux_anomaly(datatofit[0], sum_in_quad=True))
            shears.append(optmodel[0].lens_components[0].shear)
            xcen.append(optmodel[0].lens_components[0].lenstronomy_args['center_x'])
            ycen.append(optmodel[0].lens_components[0].lenstronomy_args['center_y'])
            shear_pa.append(optmodel[0].lens_components[0].shear_theta)



    #write_fluxes(filename=fluxratio_data_path+outfilename+'.txt',fluxes=np.array(flux_anomaly),mode='append')
    with open(fluxratio_data_path+'optimized_shears_LOS.txt','a') as f:
        np.savetxt(f,X=np.array(shears))
    with open(fluxratio_data_path+'optimized_shear_pa_LOS.txt','a') as f:
        np.savetxt(f,X=np.array(shear_pa))
    with open(fluxratio_data_path+'optimized_xcenter_LOS.txt','a') as f:
        np.savetxt(f,X=np.array(xcen))
    with open(fluxratio_data_path+'optimized_ycenter_LOS.txt','a') as f:
        np.savetxt(f,X=np.array(ycen))
