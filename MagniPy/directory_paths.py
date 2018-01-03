import os
import subprocess

class Paths(object):

    def __init__(self,objname=False,cuspfold='',key=False,lensdata = 'data/lensdata/',
                 gravlens_input_path='gravlens_input/',kappamap_path='gravlens_maps/',
                 temppath='temp/',initialize=True,chain=False,chain_ID=False,fname_index='',baseDIR=False,
                 sim_name='',data_4_cumu=False):


        baseDIR = os.getenv('HOME')+'/'

        if baseDIR == 'hoffman':
            baseDIR = '/u/flashscratch/g/gilmanda'+'/'

        self.baseDIR = baseDIR

        self.key=key
        self.gravlens_input_path = baseDIR+'data/'+gravlens_input_path

        self.kapmappath = self.gravlens_input_path+kappamap_path
        self.defmap_path = kappamap_path

        self.infopath = self.gravlens_input_path + 'deflector_info/'

        self.imgpos_ref = baseDIR+'data/lensdata/'+objname+'/imgpos_reference_'+cuspfold+'.txt'

        self.cuspfold=cuspfold

        self.objname=objname

        self.to_data = baseDIR+'/data/'

        if chain:
            self.gravlens_input_path_dump = self.gravlens_input_path + 'dump'+str(fname_index)+'/'
            self.datapath = baseDIR + 'data/ABC_chains/'+str(sim_name)+'/chains'+str(fname_index)+'/'
            self.lensmodel_init = self.infopath
            self.lensmodel_out = self.datapath
            self.chain_ID = chain_ID
            self.fname_index = fname_index
            self.processed_chain = baseDIR+'data/ABC_chains/processed_chains/'

            if initialize:

                self.scan_directories([self.infopath, self.datapath, self.gravlens_input_path_dump], soft=True)

        else:
            self.gravlens_input_path_dump = self.gravlens_input_path + 'dump/'
            self.temppath = self.gravlens_input_path + temppath
            self.lensmodel_init = self.infopath
            self.lensmodel_out = baseDIR + lensdata
            if objname=='':
                self.datapath = baseDIR + lensdata
            else:
                self.datapath = baseDIR + lensdata + objname +'/'
                if data_4_cumu:
                    self.datapath += 'data_4cumulative/'
                else:
                    self.datapath += 'data_4ABC/'

            if initialize:
                pass
                #self.scan_directories([self.gravlens_input_path_dump, self.infopath, self.datapath+self.key+'/', self.temppath, self.lensmodel_out, self.lensmodel_init], soft=True)

    def scan_directories(self,dirnames,soft=True):
        for name in dirnames:

            if os.path.exists(name):

                pass
            else:
                if soft==False:
                    print 'Error: create directory '+name
                    exit(1)
                #go = input('Directory: \n\n' + str(name) + '\n\n does not exist, do you want to create it (1 yes, 0 no)?  ')
                go=1
                if go == 1:
                    namesplit=name.split('/')
                    c=0
                    if len(namesplit)>=1 and type(namesplit) is list:
                        dir=namesplit[c]+'/'

                        while dir!=name:

                            while os.path.exists(dir):
                                c += 1
                                dir+=namesplit[c]+'/'


                            proc = subprocess.Popen(['mkdir', dir])
                            proc.wait()





