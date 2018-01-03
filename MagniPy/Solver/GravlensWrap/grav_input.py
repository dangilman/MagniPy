import subprocess

import numpy as np
from numpy.random import multivariate_normal as mn
import os

class gravlens_model:

    # input_models = ['alpha 1 0 0 .3 0 .3 0 0 0 1','nfw ... '] literally gravlens models; except for kappamaps
    # if want to load kapmap, set kapmap='VCC1632.lens' or other filename; set paths in 'directory_paths.py'

    def __init__(self, paths='',input_models='', kapmap='',srccoords=[],subhalos=[],
                subprofile='',shrcoords=True,outfilename=[]):

        if subhalos is not False:
            self.profile = subprofile
            self.subhalos=subhalos

        else:
            self.subhalos=subhalos
            self.profile = False

        self.kapmap = kapmap
        self.cosmology = 'LCDM'
        self.input_models = input_models

        if isinstance(srccoords, list):
            self.srccoordsx, self.srccoordsy = srccoords[0], srccoords[1]
        else:
            self.srccoordsx, self.srccoordsy = srccoords[:, 0], srccoords[:, 1]

        self.filename = paths.gravlens_input_path+'realization.in'

        self.shrcoords=False
        self.initiate(self.filename, cosmo=self.cosmology)

    def initiate(self, filename, cosmo):
        with open(filename, 'w') as f:
            if cosmo == 'LCDM':
                f.write(
                    'set omega = 0.3\nset lambda = 0.7\nset hval = 0.7\nset zlens = 0.5\nset zsrc = 1.5 \n\n')
                f.write('set omitcore = 0.001\n\n')

                if self.shrcoords:
                    f.write('set shrcoords = 1\n')

    def make(self,outfilenames=''):
        self.importkapmap = True

        for i in range(0, int(len(outfilenames))):

            if self.subhalos is not False:
                subhalos=self.subhalos[i]
            else:
                subhalos=[]

            self.write_model_params(subs=np.squeeze(subhalos),inputmod=self.input_models[i])

            if isinstance(self.srccoordsx,float):
                self.write_findimg(self.srccoordsx, self.srccoordsy, outfilenames[i])
            else:
                self.write_findimg(self.srccoordsx[i], self.srccoordsy[i], outfilenames[i])


    def write_model_params(self,critfile='',subs=False, varyflags=False, inputmod=False, load=False):

        nlens = 0
        self.kap_inmacro = False
        macro_model_string,vary_string = '',''
        if self.kapmap and self.kap_inmacro==False:

            self.kapmap,vary = self.write_macro_model(type='kappamap',model_params=self.kapmap)

            macro_model_string += '   kapmap 1 0 0 0 0 0 0 0 0 0\n'
            vary_string+=vary+'\n'
            nlens += 1
            self.kap_inmacro = True

        elif self.kapmap:
            nlens +=1

        if inputmod is not False:
            if varyflags:
                macro, vary = self.write_macro_model(model_params=inputmod, varyflags=varyflags)
                macro_model_string += macro + '\n'
                vary_string += vary[0] + '\n'
                nlens += 1
            else:
                macro, vary = self.write_macro_model(model_params=inputmod, varyflags=varyflags)
                macro_model_string += macro + '\n'
                vary_string += vary + '\n'
                nlens += 1

        else:
            count=0
            for model in range(0, len(self.input_models)):
                macro,vary = self.write_macro_model(model_params=self.input_models[model],varyflags=varyflags)
                macro_model_string+=macro+'\n'
                vary_string+=vary+'\n'
                nlens += 1
                count+=1

        if subs is not False and len(subs)>0:
            if np.array(subs).ndim==1:
                nlens+=1
            else:
                nlens+=np.shape(subs)[0]

            subinput,subvary=self.gen_sublines(subs,self.profile)

            macro_model_string+=subinput
            vary_string+=subvary

        self.write_deflector_params(self.filename, nlens, kapload=self.kapmap, macros=macro_model_string,vary=vary_string,critfile=critfile,inputmod=load)

    def write_findimg(self,xsrc,ysrc,outname):

        with open(self.filename, 'a') as f:
            if isinstance(xsrc,float):
                f.write('findimg ' + str(xsrc) + ' ' + str(ysrc) + ' ' + str(outname) + '\n')
            else:
                for i in range(0,len(xsrc)):
                    f.write('findimg ' + str(xsrc[i]) + ' ' + str(ysrc[i]) + ' ' + str(outname[i]) + '\n')
        f.close()

    def write_macro_model(self, type='canonical', model_params='',varyflags=False, *args):

        if type=='canonical':
            if varyflags:
                return '   '+model_params,[str(varyflags)]
            else:
                return '   '+model_params,'0 0 0 0 0 0 0 0 0 0'
        elif type=='kappamap':
            if self.importkapmap:
                return 'loadkapmap '+str(self.kapmap)+'\n','0 0 0 0 0 0 0 0 0 0'
            else:
                return '\n','0 0 0 0 0 0 0 0 0 0'

    def write_deflector_params(self, fname, nlens='', kapload=False, macros=False, sublines=False, vary=False, critfile=False,
                               tovary=False,inputmod=False):

        with open(fname, 'a') as f:

            if kapload and self.importkapmap:
                f.write(kapload)
                self.importkapmap=False

            if inputmod=='load':
                f.write('setlens ')
            else:
                f.write('setlens ' + str(nlens) + ' 1\n')
            if macros:
                f.write(macros)
            if sublines:
                for line in sublines[0::2]:
                    f.write(line + '\n')
            f.write(vary)
            if critfile is not False:
                if critfile!='':
                    f.write('plotcrit ' + critfile + '\n')
                else:
                    f.write(critfile)

    def gen_sublines(self,subhalos,proftype):
        lines,subvary='',''

        if proftype=='pjaffe' or proftype=='pjaffe\n':
            if subhalos.ndim==1:
                lines += '   pjaffe ' + str(subhalos[3]) + ' ' + str(subhalos[4]) + ' ' + str(
                    subhalos[5]) + ' ' + '0 0 0 0 ' + str(subhalos[2]) + ' ' + str(subhalos[1]) + ' 0\n'
                subvary += '0 0 0 0 0 0 0 0 0 0\n'
            else:
                for i in range(0,np.shape(subhalos)[0]):
                    lines+='   pjaffe '+str(subhalos[i,3])+' '+str(subhalos[i,4])+' '+str(subhalos[i,5])+' '+'0 0 0 0 '+str(subhalos[i,2])+' '+str(subhalos[i,1])+' 0\n'
                    subvary+='0 0 0 0 0 0 0 0 0 0\n'
        elif proftype=='nfw' or proftype=='nfw\n':
            if subhalos.ndim == 1:
                lines+= '   nfw ' + str(subhalos[3]) + ' ' + str(subhalos[4]) + ' ' + str(
                    subhalos[5]) + ' ' + '0 0 0 0 ' + str(subhalos[1]) + ' ' + str(0) + ' 0\n'
                subvary += '0 0 0 0 0 0 0 0 0 0\n'
            else:
                for i in range(0,np.shape(subhalos)[0]):
                    lines+='   nfw '+str(subhalos[i,3])+' '+str(subhalos[i,4])+' '+str(subhalos[i,5])+' '+'0 0 0 0 '+str(subhalos[i,1])+' '+str(0)+' 0\n'
                    subvary+='0 0 0 0 0 0 0 0 0 0\n'

        elif proftype=='tnfw' or proftype=='tnfw\n':
            if subhalos.ndim == 1:
                lines+= '   tnfw3 ' + str(subhalos[3]) + ' ' + str(subhalos[4]) + ' ' + str(
                    subhalos[5]) + ' ' + '0 0 0 0 ' + str(subhalos[2]) + ' ' + str(subhalos[1]) + ' 1\n'
                subvary += '0 0 0 0 0 0 0 0 0 0\n'
            else:
                for i in range(0,np.shape(subhalos)[0]):
                    lines+='   tnfw3 '+str(subhalos[i,3])+' '+str(subhalos[i,4])+' '+str(subhalos[i,5])+' '+'0 0 0 0 '+str(subhalos[i,2])+' '+str(subhalos[i,1])+' 1\n'
                    subvary+='0 0 0 0 0 0 0 0 0 0\n'

        elif proftype=='ptmass' or proftype=='ptmass\n':
            if subhalos.ndim == 1:
                lines+= '   ptmass ' + str(subhalos[3]) + ' ' + str(subhalos[4]) + ' ' + str(
                    subhalos[5]) + ' ' + '0 0 0 0 ' + str(0) + ' ' + str(0) + ' 0\n'
                subvary += '0 0 0 0 0 0 0 0 0 0\n'
            else:
                for i in range(0,np.shape(subhalos)[0]):
                    lines+='   ptmass '+str(subhalos[i,3])+' '+str(subhalos[i,4])+' '+str(subhalos[i,5])+' '+'0 0 0 0 '+str(0)+' '+str(0)+' 0\n'
                    subvary+='0 0 0 0 0 0 0 0 0 0\n'

        return lines,subvary

class write_inputfile:

    def __init__(self, paths, subhalos=[], src_basecoords=[], input_models=[],kapmap='',Nrealizations=1,subprofile='',**kwargs):

        self.src_basecoords = src_basecoords

        self.input_models = input_models
        self.paths=paths
        self.kapmap = kapmap

        self.subhalos=subhalos

        self.subprofile=subprofile


        self.temppath = paths.temppath
        self.dfiles = self.gen_outfilenames(Nrealizations)

        self.srccoords,self.refsrcs = self.get_coords(N=Nrealizations)

        self.shrcoords=False

        self.build()

    def build(self):

        gravlens_model(paths=self.paths,input_models=self.input_models, kapmap=self.kapmap,srccoords=self.srccoords,subhalos=self.subhalos,subprofile=self.subprofile,
                       shrcoords=self.shrcoords,outfilename=self.dfiles).make(self.dfiles)

    def get_coords(self,N):

        reference_srcs = self.scatter_src(self.src_basecoords,'imageplane',N=N)

        return reference_srcs,reference_srcs

    def import_sources(self,fnamebase):
        for i in range(1,self.Nrealizations+1):
            with open(fnamebase+str(i)+'.dat','r') as f:
                lines=f.readlines()
            f.close()
            for line in lines:
                line=line.strip().split(' ')

                if line[0]=='ptsrc':

                    if i==1:
                        srcs=np.array([float(line[2]),float(line[3])])

                    else:
                        srcs = np.vstack((srcs,np.array([float(line[2]),float(line[3])])))
        return srcs


    def scatter_src(self,basecoords, lensplane, N):

        rad = 0  # in image plane
        if N==0:
            N+=1

        if lensplane == 'imageplane':
            x0, y0 = basecoords[0], basecoords[1]

            returnarray=np.zeros((N,2))
            theta = 2 * np.pi * np.random.random(N)
            u = np.random.random(N)
            r = rad * np.sqrt(u)
            returnarray[:,0]=x0+rad*np.sqrt(u)*np.cos(theta)
            returnarray[:,1]=y0+rad*np.sqrt(u)*np.sin(theta)
            return returnarray

        elif lensplane == 'sourceplane':

            x0,y0=basecoords[:,0],basecoords[:,1]

            returnarray=np.zeros((100,2,N))

            srcsize = 5  # pc
            s = lambda x: 8698 ** -1 * (x * .5)
            cov = [[s(srcsize) ** 2, 0], [0, s(srcsize) ** 2]]

            if isinstance(x0,float):
                vals = mn([float(x0), float(y0)], cov)
                return np.array([vals[0],vals[1]])
            else:

                for j in range(0,N):
                    temp = np.array([x0[0],y0[0]])
                    for i in range(1, 100):

                        vals = mn([float(x0[j]), float(y0[j])], cov)
                        temp = np.vstack((temp, np.array([vals[0],vals[1]])))

                    returnarray[:,:,j] = temp

                return returnarray[:,:,:]

    def gen_outfilenames(self,nreal):
        dfiles = []
        for i in range(1,nreal+1):
                dfiles.append(self.temppath+'tempdata_'+str(i)+'_'+'.txt')
        return dfiles

def run_lensmod(inputfile,path_2_lensmod=False,index=False):

    if path_2_lensmod is False:
        path_2_lensmod=os.getenv('HOME')+'/'

    if path_2_lensmod=='/u/home/g/gilmanda/':
        path_2_lensmod='./lensmodel_folder/'
    elif path_2_lensmod=='/Users/danielgi/':

        if os.getcwd()=='/Users/danielgi/Code/jupyter_notebooks':
            path_2_lensmod='../'

        elif os.getcwd()=='/Users/danielgi/Code/lens_simulations/lens_simulations':
            path_2_lensmod='../../'
        elif os.getcwd()=='/Users/danielgi/Code/lens_simulations/lens_simulations/gravlens_wrapper':
            path_2_lensmod='../../../'
    else:
        #### Put the path to wherever you intall lensmodel here ####
        path_2_lensmod = os.getenv('HOME')+'/lensmodel_folder/'

    proc=subprocess.Popen([path_2_lensmod+'lensmodel',str(inputfile)])
    #proc = subprocess.Popen([path_2_lensmod+'lensmodel',''])
    proc.wait()
