import os

homedir = os.getenv('HOME')+'/'

if homedir == '/u/home/g/gilmanda/':
    prefix = '/u/flashscratch/g/gilmanda/'
    try:
        IO_directory = os.getenv('TMPDIR') + '/'
    except:
        IO_directory = os.getenv('SCRATCH')+ '/'

    lensmodel_location = homedir+'/Code/lensmodel_folder/'
    path_2_lensmodel = IO_directory
    gravlens_input_path = IO_directory + 'gravlens_input/'

elif homedir == '/Users/danielgilman/':
    prefix = homedir
    IO_directory = prefix
    path_2_lensmodel = homedir + '/Code/'
    gravlens_input_path = IO_directory + 'data/gravlens_input/'

elif homedir == '/home/users/sibirrer/':

    # wherever the file storage directory is for the stanford cluster.
    prefix = '/scratch/users/sibirrer/'
    # insde the directory 'prefix', create the directory structure ./data/ABC_chains/
    # where the readout will happen

    # ignore these
    IO_directory = prefix
    lensmodel_location = homedir + '/Code/lensmodel_folder/'
    path_2_lensmodel = IO_directory
    gravlens_input_path = IO_directory + 'gravlens_input/'

elif homedir == '/home/nierenbe/':

    # wherever the file storage directory is for the JPL cluster.
    prefix = '/aurora_nobackup/abclens/'

    # can make this the same as prefix
    IO_directory = prefix

    # don't need these ones
    lensmodel_location = homedir + '/Code/lensmodel_folder/'
    path_2_lensmodel = IO_directory

    # don't need this one either
    gravlens_input_path = IO_directory + 'gravlens_input/'

elif homedir == '/home/gilmanda/':
    prefix = homedir
    IO_directory = prefix
    path_2_lensmodel = homedir
    gravlens_input_path = IO_directory + 'data/gravlens_input/'

# change this to your home directory on the cluster
elif homedir == 'benson/cluster/homedirectory':

    # enter the path to the storage space on the cluster
    # insde the directory prefix, create the directories ./data/ABC_chains/
    prefix = '/path_to/storage/oncluster/'

    IO_directory = prefix

    # ignore these lines
    lensmodel_location = homedir + '/'
    path_2_lensmodel = homedir + '/'
    gravlens_input_path = IO_directory + 'gravlens_input/'

else:
    raise Exception('You seem to be using a different computer. The home directory '+str(homedir)+' is not found.')

#if os.path.exists(path_2_lensmodel+'lensmodel'):
#    pass
#else:
#    shutil.copy2(lensmodel_location+'lensmodel',path_2_lensmodel)

kapmappath = gravlens_input_path+'gravlens_maps/'

defmap_path = kapmappath

infopath = gravlens_input_path + 'deflector_info/'

data_path = prefix+'data/lensdata/SIE_data/data_4ABC/'

gravlens_input_path_dump = gravlens_input_path + 'dump/'

# Need to create the directories 'data' and 'data/ABC_chains/' in the storage directory
chainpath = prefix + 'data/ABC_chains/'

chainpath_out = prefix + 'data/sims/'

fluxratio_data_path = prefix+'data/fluxratio_distributions/'

chain_output_path = prefix+'/data/ABC_chains/'
