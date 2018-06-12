import os
import shutil

homedir = os.getenv('HOME')+'/'

if homedir == '/u/home/g/gilmanda/':
    prefix = '/u/flashscratch/g/gilmanda/'

    #try:
    #    IO_directory = os.getenv('TMPDIR') + '/'
    #except:
    #    IO_directory = os.getenv('NEWSCRATCH')+ '/'

    IO_directory = os.getenv('NEWSCRATCH')+ '/'

    lensmodel_location = homedir+'/Code/lensmodel_folder/'
    path_2_lensmodel = IO_directory
    gravlens_input_path = IO_directory + 'gravlens_input/'

elif homedir == '/Users/danielgi/':
    prefix = homedir
    IO_directory = prefix
    path_2_lensmodel = homedir + '/Code/'
    gravlens_input_path = IO_directory + 'data/gravlens_input/'

elif homedir == '/Users/mcsedarous/':
    prefix = homedir
    IO_directory = prefix
    path_2_lensmodel = homedir + 'Desktop/research_Treu/'
    gravlens_input_path = IO_directory + 'data/gravlens_input/'

if os.path.exists(path_2_lensmodel+'lensmodel'):
    pass
else:
    shutil.copy2(lensmodel_location+'lensmodel',path_2_lensmodel)

kapmappath = gravlens_input_path+'gravlens_maps/'

defmap_path = kapmappath

infopath = gravlens_input_path + 'deflector_info/'

data_path = prefix+'data/lensdata/SIE_data/data_4ABC/'

gravlens_input_path_dump = gravlens_input_path + 'dump/'

chainpath = prefix + 'data/ABC_chains/'

fluxratio_data_path = prefix+'data/fluxratio_distributions/'

chain_output_path = prefix+'/data/ABC_chains/'
