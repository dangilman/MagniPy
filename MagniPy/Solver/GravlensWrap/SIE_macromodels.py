import os
from run_lensmodel import run_lensmod
from lens_simulations.DM_generator.dm_realization import *
from lens_simulations.lens_routines.lens_mapping import get_SIEmags
from lens_simulations.lens_routines.lens_classify import *
from gravlens_interface import *

N = 100
N_src_sample = 40000

import sys

start = str(sys.argv[1])
norm = str(sys.argv[2])
profile = 'tnfw'
if start=='CDM':
    dm_model = start+'_'+norm+'_'+profile+'_1_6_10'
else:
    c = sys.argv[4]
    dm_model = start + '_' + norm + '_'+profile+'_'+str(c)+'_6_10_'+str(sys.argv[3])

macro_type = 'SIE'

cuspdata = ''
folddata = ''
crossdata = ''
ncusp,nfold,ncross = 0, 0,0
astrometric_error = 0.003 #mas

bvals = np.random.normal(1,.2,N)
bvals = np.absolute(bvals)
ellip = np.random.normal(.2,.05,N)
ellip = np.absolute(ellip)
shear = np.random.normal(0.05,.01,N)
shear = np.absolute(shear)
ePA = np.random.uniform(-90,90,N)
sPA = np.random.uniform(-90,90,N)
#print bvals
#exit(1)

baseDIR = os.getenv('HOME') + '/'
path_to_output = baseDIR + 'data/gravlens_input/temp2/'

fnamelensmod = path_to_output+'lensmodelin.in'
dpath = os.getenv('HOME')+'/data/gravlens_input/temp2/'
finish = False

while finish is False:

    subhalos=[]
    with open(fnamelensmod,'w') as f:
        f.write('set omega=0.3\nset lambda=0.7\nset hval=0.7\nset zlens=0.5\nset zsrc=1.5\nset omitcore=0.001\nset shrcoords = 2\n')

        for i in range(0,N):
            DM = DM_realization(Rein=bvals[i])
            DM.set_sub_params(dm_model)
            subs = DM.draw_subhalos(N=1)[0]
            subhalos.append(subs)
            Nsub = DM.Nsub

            if Nsub==1:
                subs = subs[0]
                subs = subs[np.newaxis,:]

            Nmodel = Nsub+1

            f.write('setlens '+str(Nmodel)+' 1\n')

            if macro_type=='SIE':
                f.write('   alpha '+str(bvals[i])+' '+'0 0 '+str(ellip[i])+' '+str(ePA[i])+' '+str(shear[i])+' '+str(sPA[i])+' 0 0 '+str(1)+'\n')
            elif macro_type=='SNFW':
                f.write('   sersic ' + str(normsers[i]) + ' ' + '0 0 ' + str(ellip[i]) + ' ' + str(ePA[i]) + ' 0 0 '+str(re[i])+' 0 '+str(n)+'\n')
                f.write('   nfw ' + str(ks[i]) + ' 0 0 0 0 '+str(shear[i])+' '+str(sPA[i])+' '+str(rs[i]) + ' 0 0\n')

            if profile=='pjaffe':
                for j in range(0,Nsub):
                    f.write('   pjaffe '+str(subs[j,3])+' '+str(subs[j,4])+' '+str(subs[j,5])+' 0 0 0 0 '+str(subs[j,2])+' '+str(subs[j,1])+' 0\n')
                for j in range(0,Nmodel):
                    f.write('0 0 0 0 0 0 0 0 0 0\n')
            elif profile=='nfw':
                for j in range(0,Nsub):
                    f.write('   nfw '+str(subs[j,3])+' '+str(subs[j,4])+' '+str(subs[j,5])+' 0 0 0 0 '+str(subs[j,1])+' 0 0\n')
                for j in range(0,Nmodel):
                    f.write('0 0 0 0 0 0 0 0 0 0\n')
            elif profile=='tnfw':
                for j in range(0,Nsub):
                    f.write('   tnfw3 '+str(subs[j,3])+' '+str(subs[j,4])+' '+str(subs[j,5])+' 0 0 0 0 '+str(subs[j,2])+' '+str(subs[j,1])+' 1\n')
                for j in range(0,Nmodel):
                    f.write('0 0 0 0 0 0 0 0 0 0\n')

            f.write('mock1 '+path_to_output+'SIEtest'+str(i+1)+' '+str(N_src_sample)+' 4\n')
        f.close()
    #a=input('continue')
    run_lensmod(path_to_output + 'lensmodelin.in')
    cusp_data,fold_data,cross_data = [],[],[]

    for i in range(0,N):
        cross,cusp,fold = False,False,False
        ximg,yimg,tdel,src = readimages(path_to_output + 'SIEtest' + str(i + 1))

        if ximg is False or yimg is False:
            continue

        for k in range(0,np.shape(ximg)[0]):
            try:
                x,y,t = ximg[k,:],yimg[k,:],tdel[k,:]
            except:

                x,y,t = ximg[:,np.newaxis],yimg[:,np.newaxis],tdel[:,np.newaxis]

            sortinds = np.argsort(t)
            t = t[sortinds]
            x = x[sortinds]
            y = y[sortinds]

            dx=np.random.normal(0,astrometric_error,4)
            dy=np.random.normal(0,astrometric_error,4)

            x = np.array(x)+dx
            y = np.array(y)+dy

            code = identify(x,y,bvals[i])


            submodel = [['alpha', bvals[i], '0', '0', ellip[i], ePA[i], shear[i], sPA[i], '0', '0', '1']]

            if code==0:
                cross = True
                if ncross<N:
                    mag = get_SIEmags(x, y, [subhalos[i]],submodel,src[k], fluxerr=0, src_size=0.0012, subprofile=profile,shrcoords=2)
                    crossdata += '4 ' + str(src[k][0]) + ' ' + str(src[k][1])
                    for z in range(0, 4):
                        crossdata += ' ' + str(x[z]) + ' ' + str(y[z]) + ' ' + str(mag[z]) + ' ' + str(t[z])
                    crossdata += '\n'
                    ncross+=1

            elif code==1:
                fold = True
                if nfold<N:
                    mag = get_SIEmags(x, y, [subhalos[i]], submodel, src[k], fluxerr=0, src_size=0.0012,
                                      subprofile=profile,shrcoords=2)
                    folddata += '4 ' + str(src[k][0]) + ' ' + str(src[k][1])
                    for z in range(0, 4):
                        folddata += ' ' + str(x[z]) + ' ' + str(y[z]) + ' ' + str(mag[z]) + ' ' + str(t[z])
                    folddata += '\n'
                    nfold+=1

            else:
                cusp = True
                if ncusp<N:
                    mag = get_SIEmags(x, y, [subhalos[i]], submodel, src[k], fluxerr=0, src_size=0.0012,subprofile=profile, shrcoords=2)
                    cuspdata += '4 ' + str(src[k][0]) + ' ' + str(src[k][1])
                    for z in range(0,4):
                        cuspdata+=' '+str(x[z])+' '+str(y[z])+' '+str(mag[z])+' '+str(t[z])
                    cuspdata+='\n'
                    ncusp+=1


            if cross==fold and cusp==fold and fold == True:

                break

        if ncusp==N and nfold==N and ncross==N:
            finish=True

with open(os.getenv('HOME')+'/data/lensdata/SIE_data/'+dm_model+'/cusp_data.txt','a') as f:
    f.write(cuspdata)
f.close()
with open(os.getenv('HOME')+'/data/lensdata/SIE_data/'+dm_model+'/fold_data.txt','a') as f:
    f.write(folddata)
f.close()
with open(os.getenv('HOME')+'/data/lensdata/SIE_data/'+dm_model+'/cross_data.txt','a') as f:
    f.write(crossdata)
f.close()