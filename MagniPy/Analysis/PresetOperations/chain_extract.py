from lens_simulations.statistical_tools.ChainAssemble import BuildChain
from lens_simulations.statistical_tools.AbcChain import Chain
from lens_simulations.directory_paths import Paths
import numpy as np
import os

baseDIR = 'hoffman'

N_lenses = 10

Nchains = 106 * N_lenses  # 50 lenses

img_configs = ['cusp', 'fold', 'cross'] * 3+['cusp']

assert len(img_configs) == N_lenses

objnames = ['SIEdata'] * N_lenses

Nsamples = 20
d1, d2 = 20, 20
step = Nchains / len(objnames)
chains_perlens = step

input_names = ['gamma_2.08']

folder = input_names[0] + '/'

counter = 0
start_index = 0

for i in range(start_index, N_lenses + start_index):

    print 'extracting lens ' + str(i + 1) + ': ' + str(objnames[counter]) + ' (' + str(img_configs[counter]) + ')'

    inds_tokeep = np.arange(i * step + 1, (i + 1) * step + 1)
    folder = input_names[0] + '/'
    simulation_name = folder + 'chain_' + str(i + 1)

    fnames = {'fluxfile': simulation_name + '_fluxes.txt', 'paramfile': simulation_name + '_params.txt',
              'infofile': simulation_name + '_info.txt'}

    chain = BuildChain(objnames=objnames, sim_names=input_names, img_configs=img_configs, indmax=Nchains,
                       chains_perlens=chains_perlens, baseDIR=baseDIR, Nsamples=Nsamples, d1=d1, d2=d2)

    chain.read_chain(inds_tokeep)

    chain.write_chain_files(fnames=fnames, nkeep=None, noriginal=2800)

    counter += 1
