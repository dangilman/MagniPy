from MagniPy.ABCsampler.process import bootstrap_intervals
import numpy as np


def bootstrap(chain_name, Nlenses, error, which_lenses, Nbootstraps, fname_prefix):

    interval = bootstrap_intervals(chain_name, Nlenses, which_lenses,
               'log_m_break', Nbootstraps, error, 1500)
    nlens = interval['Nlenses']
    high95 = interval['high_95']

    for ni, hi in zip(nlens, high95):

        with open(fname_prefix+'_'+str(ni)+'lenses_2sigma_error'+str(error)+'.txt','a') as f:
            f.write(str(hi)+'\n')

def run():

    for error in [0.04,0.08]:
        for rep in range(0, 2):
            chain_names = ['CDM_sigma0.023_srcsize0.035','CDM_sigma0.012_srcsize0.03']
            path_prefix = ['highnorm', 'lownorm']
            for i, chain in enumerate(chain_names):
                N_lenses = [10, 15, 20, 25, 30, 35, 40, 45]
                bootstrap(chain, N_lenses, error, np.arange(1, 47), Nbootstraps=1, fname_prefix=path_prefix[i])

run()