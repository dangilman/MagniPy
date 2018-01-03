import lens_simulations.gravlens_wrapper.run_lens

gen_realizations = True
fit_realizations = True

baseDIR = 'amedei'
input_models = 'load deflector'

objnames=['VCC731']
#objnames=['VCC731']
source_size = 0.0012 #10 parcsecs
gensubhalos=True
subprofile = 'tnfw'

mlow,mhigh = 6,10

measurement_error = 0 # m.a.s.
fluxerror = 0 # five percent
data_4_cumu = True
flux_sigma = 0.7

cusp_or_fold=['cusp']
c=1
# 1 means concentration turns over, 0 no

filter_spatial,mindis = False,[.5,.75,.75,.75]

key = ['WDM_.0025_'+str(subprofile)+'_'+str(c)+'_'+str(mlow)+'_'+str(mhigh)+'_9_mindis.5m8']

Nrealizations=1000

for objname in objnames:
    kapmap = objname + '.lens'
    for cuspfold in cusp_or_fold:

        if gen_realizations:
            for i in range(0,len(key)):

                if key[i]=='nosub_':
                    gensubhalos=False
                else:
                    gensubhalos=True

                if key[i]=='nosub_':
                    inkey=key[i]+cuspfold
                else:
                    inkey = key[i]


                lens_simulations.gravlens_wrapper.run_lens.run_lenssim(input_models=input_models, objname=objname, key=inkey,
                                                                       gensubhalos=gensubhalos,cuspfold=cuspfold,
                                                                       Nrealizations=Nrealizations, extendedsrc=True,
                                                                       kapmap=kapmap, srcsize=source_size, position_sigma=measurement_error,
                                                                       fluxerror=fluxerror,baseDIR=baseDIR,
                                                                       data_4_cumu=data_4_cumu,filter_spatial=filter_spatial,mindis=mindis[i])
                    #if do_over_SIE:
                    #    run_lens.run_lenssim(input_models='SIEfit',objname=objname,key=key[i],gensubhalos='from file',cuspfold=cuspfold,Nrealizations=Nrealizations,extendedsrc=False)
    for cuspfold in cusp_or_fold:
        if fit_realizations:
            for i in range(0,len(key)):
                if key[i]=='nosub_':
                    inkey=key[i]+cuspfold
                else:
                    inkey = key[i]
                extras = ['']
                #fit_type,Nrealizations = 0,1
                fit_type=1
                lens_simulations.gravlens_wrapper.run_lensmodel.run_lensmodel(objname=objname, key=inkey,
                                                                              cuspfold=cuspfold, fit_type=fit_type,
                                                                              Nfiles=Nrealizations, extras=extras,
                                                                              baseDIR = baseDIR, data_4_cumu=data_4_cumu,
                                                                              flux_sigma=flux_sigma)
