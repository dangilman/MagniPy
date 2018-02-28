import os

homedir = os.getenv('HOME')+'/'

if homedir == '/u/home/g/gilmanda':
    prefix = '/u/flashscratch/g/gilmanda/'
    path_2_lensmodel = homedir+'/Code/'

elif homedir == '/Users/danielgi/':
    prefix = homedir
    path_2_lensmodel = homedir + '/Code/'

elif homedir == '/Users/mcsedarous/':
    prefix = homedir
    path_2_lensmodel = homedir + 'Desktop/research_Treu/'

gravlens_input_path = prefix+'data/gravlens_input/'

kapmappath = gravlens_input_path+'gravlens_maps/'

defmap_path = kapmappath

infopath = gravlens_input_path + 'deflector_info/'

imgpos_ref = prefix+'data/lensdata/'

gravlens_input_path_dump = gravlens_input_path + 'dump/'

chainpath = prefix + 'data/ABC_chains/'

fluxratio_data_path = prefix+'data/lensdata/SIE_flux_ratios/'
