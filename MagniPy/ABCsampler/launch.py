import sys
from MagniPy.ABCsampler.ABCsampler import runABC
import time
from MagniPy.paths import *

start_time = time.time()
job_index = int(sys.argv[1])

# edit the chain_ID to specify the folder containing the 'paramdictionary' files for each simulation.
# the param dictionary files should be in a folder located in the 'prefix' directory, see "paths.py"
chain_ID = 'WDM_run_7.7'

#time.sleep(200)
runABC(prefix+chain_ID+'/',job_index)

print("--- %s seconds ---" % (time.time() - start_time))