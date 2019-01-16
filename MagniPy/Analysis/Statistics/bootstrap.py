from MagniPy.ABCsampler.process import bootstrap_intervals
import numpy as np


def bootstrap(chain_name, Nlenses, error, which_lenses, Nbootstraps, fname_prefix):

    interval = bootstrap_intervals(chain_name, Nlenses, which_lenses,
                                   Nbootstraps, error, 2000, bandwidth_scale=0.6, bins=4)
    nlens = interval['Nlenses']
    high95 = interval['high_95']

    for ni, hi in zip(nlens, high95):

        with open(fname_prefix+'_'+str(ni)+'lenses_2sigma_error'+str(error)+'.txt','a') as f:
            f.write(str(hi)+'\n')

def run():

    #N_lenses = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    N_lenses = [2,4]
    for k in range(1, 2):
        for error in [0.02, 0.04, 0.06, 0.08]:
            for rep in range(0, 1):
                chain_names = ['CDM_sigma1.6_srcsize30','CDM_sigma0.025_srcsize0.033']
                path_prefix = ['lownorm', 'highnorm']
                for i, chain in enumerate(chain_names):

                    bootstrap(chain, N_lenses, error, np.arange(1, 51), Nbootstraps=1, fname_prefix=path_prefix[i])

run()