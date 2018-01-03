from lens_simulations.statistical_tools.execute_ABC import *
import os
import time
import sys

def read_datafile(dfilein):
    x,y,f,t = np.loadtxt(dfilein,unpack=True)
    return [x,y,f,t]

def read_paraminput(file):
    with open(file,'r') as f:
        vals = f.read()
    return eval(vals)

#print sys.argv[1],sys.argv[2]
#job_indexs = np.arange(int(sys.argv[1]),int(sys.argv[2]))

job_indexs=[7430]
for job_index in job_indexs:
    start_time = time.time()
    path_2_input = os.getenv('HOME')+'/data/hoffman_input/'
    pfile = path_2_input+'paramdictionary'+'_jobID'+str(job_index)+'.txt'
    params = read_paraminput(pfile)

    dfile_ind = params['data_index']
    dataarray = read_datafile(dfilein=path_2_input+'data'+str(dfile_ind)+'.txt')

    # Don't worry about reading in a data file, just use this

    params['dataarray'] = dataarray

    # this sets the relative paths; will default to your home directory
    params['baseDIR']='amedei'

    ##### Launch the program #####
    if os.path.exists(os.getenv('HOME')+'/data/ABC_chains/' + params['sim_name'] + '/chains' + str(job_index) + '/chain.txt'):
        continue
    else:
        run_abc(**params)
        print("--- %s seconds ---" % (time.time() - start_time))
