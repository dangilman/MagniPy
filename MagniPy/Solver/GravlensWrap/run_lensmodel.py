import shutil
from grav_input import run_lensmod
from lens_simulations.data_visualization.data_handling import *
from lens_simulations.lens_routines.lens_mapping import *


def add_endings(extra_constraints, fname):
    if extra_constraints is not False:
        if 'fitflux' in extra_constraints:
            fname += '_fitflux'
        elif 'fixshear' in extra_constraints:
            fname += '_fixshear'
        elif 'withsub' in extra_constraints:
            fname += '_withsub'
    return fname

def parse_to_lensmodel(subs,prof,nsub):

    if prof=='pjaffe':
        if nsub==1:
            return prof,[subs[3]],[subs[4]],[subs[5]],[subs[2]],[subs[1]],[0]
        else:
            return prof,subs[:,3],subs[:,4],subs[:,5],subs[:,2],subs[:,1],[0]*nsub
    elif prof=='nfw':
        if nsub==1:
            return prof,[subs[3]],[subs[4]],[subs[5]],[subs[1]],[0],[0]
        else:
            return prof,subs[:,3],subs[:,4],subs[:,5],subs[:,1],[0]*nsub,[0]*nsub
    elif prof=='tnfw':
        if nsub==1:
            return prof+str(3),[subs[3]],[subs[4]],[subs[5]],[subs[2]],[subs[1]],[1]
        else:
            return prof+str(3),subs[:,3],subs[:,4],subs[:,5],subs[:,2],subs[:,1],[1]*nsub

    else:
        raise ValueError('specify subprofile')


def make_inputs(startmod,dfile_name,outfile,lensmod_commandfile,fit_type=1,open_type='a',
                writecosmo=True,write_setgrid=True,**kwargs):
    inputstring=''
    frontspace=''
    if fit_type == 0:
        inputstring = 'set omega = 0.3\nset lambda = 0.7\nset hval = 0.7\nset zlens = 0.5\nset zsrc = 1.5\nset shrcoords=1\nset omitcore=.001\nset checkparity=0\n'
        inputstring += 'data ' + dfile_name + '\n'
        fitoutfile1 = kwargs['outfile_init']
        inputstring += 'set gridflag = 0\nset chimode = 0\nset restart = 3\n\n'
        inputstring += 'setlens 1 1\n'+frontspace+'alpha 1 0 0 0 0 0 0 0 0 1\n1 1 1 1 1 1 1 0 0 0\n'
        inputstring += 'randomize 5 ' + fitoutfile1 + '\n'
        inputstring += '0 1.2\n-0.1 0.1\n-0.1 0.1\n-0.1 0.1\n-0.1 0.1\n-0.1 0.1\n-0.1 0.1\n'
        inputstring += 'set gridflag = 1\nset chimode = 1\n'
        inputstring += 'setlens ' + fitoutfile1 + '.start\n'
        inputstring += 'optimize ' + outfile

    elif fit_type == 1:
        if writecosmo:
            inputstring = 'set omega = 0.3\nset lambda = 0.7\nset hval = 0.7\nset zlens = 0.5\nset zsrc = 1.5\nset shrcoords=1\nset omitcore=.001\nset checkparity=0\n'
        else:
            inputstring=''
        if write_setgrid:
            inputstring += 'set gridflag = 0\nset chimode = 0\nset restart = 1\n\n'

        inputstring += 'data ' + dfile_name + '\n'
        inputstring += 'setlens 1 1\n' + frontspace + str(startmod) + '\n1 1 1 1 1 1 1 0 0 0\n'
        inputstring += 'optimize '+outfile+'\n'

    elif fit_type == 2:  # optimize with subhalos

        subs = kwargs['subhalos']

        if writecosmo:

            inputstring = 'set omega = 0.3\nset lambda = 0.7\nset hval = 0.7\nset zlens = 0.5\nset zsrc = 1.5\nset shrcoords=1\nset omitcore=.001\nset checkparity=0\n'

        else:

            inputstring = ''

        inputstring += 'data ' + dfile_name + '\n'

        inputstring += 'set gridflag = 0\nset chimode = 0\nset restart = 1\n\n'

        for n in range(0,len(subs)):

            subrealization = np.squeeze(subs[n])

            nlens = 1

            if np.shape(subrealization)[0]==0:
                nsub=0
            else:
                if np.squeeze(subrealization).ndim==1:
                    nsub = 1
                else:
                    nsub = np.shape(subrealization)[0]

            nlens += nsub

            inputstring += 'setlens ' + str(nsub + 1) + ' 1\n'

            inputstring += frontspace + str(startmod) + '\n'

            vary = '1 1 1 1 1 1 1 0 0 0\n'

            if nsub > 0:

                prof, norm, x, y, p8, p9, p10 = parse_to_lensmodel(subrealization, kwargs['subprofile'], nsub)

                for s in range(0, nsub):

                    vary += '0 0 0 0 0 0 0 0 0 0\n'

                    inputstring += frontspace + prof + ' ' + str(norm[s]) + ' ' + str(x[s]) + ' ' + str(y[s]) + ' 0 0 0 0 ' + str(p8[s]) + ' ' + str(p9[s]) + ' '+str(p10[s])+'\n'


            inputstring += vary

            inputstring += 'optimize ' + outfile + str(n + 1) + '\n'

    with open(lensmod_commandfile, open_type) as f:

        f.write(inputstring)

    f.close()


def write_data_file(fname, xpos, ypos, mag, tdel, extras=False, fit_type=0, constrainflux=True, flux_sigma = 0.5,
                    tdel_sigma=2,position_sigma=0.003,**kwargs):

    if extras is not False:

        if 'fitflux' in extras:
            constrainflux = True

    if fit_type==2:
        constrainflux=True
        if 'fitflux' in extras:
            pass
        else:
            extras.append('fitflux')

    maxflux = np.max(mag)
    maxtdel = np.max(tdel)

    with open(fname, 'a') as f:

        f.write(str(1) + '\n')
        f.write('0.000000e+00 0.000000e+00 40.000000e-03\n')
        f.write(str(0.0) + ' ' + str(10000) + '\n')
        f.write(str(0.0) + ' ' + str(10000) + '\n')
        f.write(str(0.0) + ' ' + str(10000) + '\n')
        f.write(str(1) + '\n' + str(4) + '\n')

        tdelsig = .02

        for i in range(0, 4):

            if constrainflux:
                f.write(str(xpos[i]) + ' ' + str(ypos[i]) + ' ' + str(
                    float(mag[i]) * maxflux ** -1) + ' ' + str(position_sigma) + ' ' + str(
                    flux_sigma * float(mag[i]) * maxflux ** -1) + ' ' + str(tdel[i]) + ' ' + str(tdelsig) + '\n')

            else:
                f.write(str(xpos[i]) + ' ' + str(ypos[i]) + ' ' + str(mag[i]*maxflux**-1) + ' ' + str(
                    position_sigma) + ' ' + str(1) + ' ' + str(tdel[i]) + ' ' + str(tdelsig) + '\n')
            tdelsig = tdel_sigma
    f.close()

def add_macroline(fname_input,line,nmods):
    with open(fname_input,'a') as f:
        f.write(line)
    f.close()

def read_write_data(fit_name, out_datafile,readonly=False,read_single=False):

    x_srcSIE,y_srcSIE=[],[]

    with open(fit_name,'r') as f:

        nextline = False
        dosrc = False
        doimg = False
        count=0
        readcount = 0

        for line in f:
            row=line.split(" ")
            row_split=filter(None,row)
            if row_split[0]=='alpha':
                macromodel=row_split

                continue

            if row_split[0]=='Source':

                nextline=True
                dosrc=True
                src = []
                continue

            if nextline and dosrc:

                for item in row:
                    try:
                        src.append(float(item))
                    except ValueError:
                        continue
                x_srcSIE.append(src[0])
                y_srcSIE.append(src[1])
                nextline=False
                dosrc=False
                continue

            if row_split[0]=='images:\n':
                nextline=True
                doimg=True
                count=0
                x, y, f, t = [], [], [], []
                continue

            if nextline and doimg:

                count+=1
                numbers = []
                for item in row:
                    try:
                        numbers.append(float(item))
                    except ValueError:
                        continue
                x.append(numbers[4])
                y.append(numbers[5])
                f.append(numbers[6])
                t.append(numbers[7])

                if count == 4:
                    t = np.array(t)

                    if min(t) < 0:
                        t += -1 * min(t)


                    if readcount==0:
                        xpos = x
                        ypos = y
                        fr = np.array(f)*max(np.array(f))**-1
                        tdel=np.array(t)
                        if read_single:

                            return xpos,ypos,fr,t,macromodel,[x_srcSIE[0],y_srcSIE[0]]

                    else:
                        xpos = np.vstack((xpos,x))
                        ypos = np.vstack((ypos,y))
                        fr = np.vstack((fr,f))
                        tdel = np.vstack((tdel,t))
                    readcount+=1

                    nextline=False
                    doimg=False
                    continue

    if readonly:

        return xpos,ypos,fr,t,macromodel,[x_srcSIE[0],y_srcSIE[0]]

    else:

        with open(out_datafile, 'a') as g:

            for k in range(0,np.shape(xpos)[0]):
                g.write('4 ' + str(x_srcSIE[k]) + ' ' + str(y_srcSIE[k]))

                for i in range(0, 4):
                    g.write(
                        ' ' + str(xpos[k,i]) + ' ' + str(ypos[k,i]) + ' ' + str(fr[k,i] * np.max(fr[k,:]) ** -1) + ' ' + str(
                            tdel[k,i]))
                g.write('\n')
        g.close()


def run_lensmodel(path=False,objname='', key=[], cuspfold='', fit_type=1, Nfiles=int, subhalos=False, Nstart=0,data_4_cumu=False,
                  flux_sigma=0.5,**kwargs):
    # read in data to fit

    if path is False:
        if 'open_chain' in kwargs:
            path = directory_paths.Paths(objname=objname, key=key, cuspfold=cuspfold, baseDIR=kwargs['baseDIR'], chain=True)
        else:
            path = directory_paths.Paths(objname=objname, key=key, cuspfold=cuspfold, baseDIR=kwargs['baseDIR'],data_4_cumu=data_4_cumu)

    #init_name = path.lensmodel_init+path.objname+'_'+path.cuspfold+'_SIEstart.dat'

    lensmod_infile = path.gravlens_input_path_dump+'opt.in'

    DATA = read_data(paths=path, **kwargs)

    xpos_tofit, ypos_tofit, mag_tofit, tdel_tofit = DATA.readinData(dfile_key=key, returnfull=True, **kwargs)

    final_outfile = path.datapath + key +'/'+ path.cuspfold + '_SIEfit_data.txt'

    dfilename = path.gravlens_input_path_dump + 'data_in_' + str(1) + '.txt'
    dumppath = path.gravlens_input_path_dump + 'init_' + str(1)
    out = path.gravlens_input_path_dump + 'bf_SIE_' + path.cuspfold + '_' + str(1)

    write_data_file(fname=dfilename, xpos=xpos_tofit[0, :],
                        ypos=ypos_tofit[0, :], mag=mag_tofit[0, :],
                        tdel=tdel_tofit[0, :], fit_type=0)

    make_inputs(startmod='', dfile_name=dfilename,outfile_init=dumppath, outfile=out,
                lensmod_commandfile=lensmod_infile,fit_type=0, open_type='w')

    run_lensmod(path.gravlens_input_path_dump + 'opt.in')
    startmod = import_deflector_params(out + '.dat')

    with open(path.gravlens_input_path_dump + 'opt.in', 'w') as f:
        pass
    f.close()
    out_toread = []
    for i in range(1,Nfiles+1):
        dfilename = path.gravlens_input_path_dump + 'data'+str(i)+'.txt'
        with open(dfilename, 'w') as f:
            pass
        f.close()

        write_data_file(fname=dfilename, xpos=xpos_tofit[Nstart + i - 1, :],
                        ypos=ypos_tofit[Nstart + i - 1, :], mag=mag_tofit[Nstart + i - 1, :],
                        tdel=tdel_tofit[Nstart + i - 1, :], fit_type=fit_type, constrainflux=True,flux_sigma=flux_sigma)
        if i==1:
            writecosmo=True
            write_setgrid=True
        else:
            writecosmo=False
            write_setgrid=False

        out = path.gravlens_input_path_dump + 'fit_'+str(i)
        out_toread.append(out+'.dat')
        make_inputs(startmod=startmod, dfile_name=dfilename,outfile=out,lensmod_commandfile=path.gravlens_input_path_dump + 'opt.in',
                    fit_type=1,writecosmo=writecosmo,write_setgrid=write_setgrid,Ntofit=Nfiles+1)

    run_lensmod(path.gravlens_input_path_dump + 'opt.in')

    with open(final_outfile,'w') as f:
        pass
    f.close()

    for i in range(1,Nfiles+1):

        x_fit,y_fit,mag_fit,t_fit,SIE_mod,src_fit = read_write_data(out_toread[i-1], '', readonly=True,read_single=True)
        if i>1:
            xfit = np.vstack((xfit,x_fit))
            yfit = np.vstack((yfit,y_fit))
            magfit = np.vstack((magfit,mag_fit))
            tfit = np.vstack((tfit,t_fit))
            SIEmod.append(SIE_mod)
            srcfit = np.vstack((srcfit,np.array(src_fit)))
        else:
            xfit,yfit,magfit,tfit,SIEmod,srcfit = x_fit,y_fit,mag_fit,t_fit,[SIE_mod],src_fit

    fname = final_outfile
    subs = [[] for i in range(0,np.shape(xfit)[0])]

    mag_final = get_SIEmags(xfit,yfit,subs,SIEmod,srcfit)

    write_files(fname, x=xfit, y=yfit, m=mag_final, t=tfit, srcpos=srcfit)
    #read_write_data(readfile, final_outfile, readonly=False)

    shutil.rmtree(path.gravlens_input_path_dump)

def do_fit(dfilename,outpath,outname,inname,path,x2fit,y2fit,mag2fit,tdel2fit,fit_type,writecosmo=False,
           write_setgrid=False,startmod='',subhalos=False,subprof='',constrainflux=False,flux_sigma=0.1):

    write_data_file(fname=dfilename, xpos=x2fit,ypos=y2fit, mag=mag2fit,tdel=tdel2fit, fit_type=1, constrainflux=constrainflux, flux_sigma=flux_sigma)

    make_inputs(startmod=startmod, dfile_name=dfilename, outfile_init=outpath, outfile=outname,
                lensmod_commandfile=inname, fit_type=fit_type, open_type='w',writecosmo=writecosmo,write_setgrid=write_setgrid,
                subhalos=subhalos,subprofile=subprof)

    run_lensmod(path.gravlens_input_path_dump + 'opt.in')

def init(path_out):
    with open(path_out + 'opt.in', 'w') as f:
        pass
    f.close()

    dfilename = path_out + 'batch_data.txt'
    with open(dfilename, 'w') as f:
        pass
    f.close()

def run_chainfit(path=classmethod, xpos_tofit=[], ypos_tofit=[], mag_tofit=[], tdel_tofit=[], subhalos=[],
                 Nsims=int, subprof='',dfileind=int, task_index=int,flux_sigma=0.2):
    # fit the 'real' lens data, get the starting SIE model

    init(path.gravlens_input_path_dump)

    dfilename = path.gravlens_input_path_dump + 'data_in_' + str(dfileind) + '.txt'
    dumppath = path.gravlens_input_path_dump + 'init_' + str(task_index)
    out = path.gravlens_input_path_dump + 'real_SIE_' + path.cuspfold

    lensmod_infile = path.gravlens_input_path_dump + 'opt.in'

    do_fit(dfilename, dumppath, out, lensmod_infile, path,np.squeeze(xpos_tofit), np.squeeze(ypos_tofit),
           np.squeeze(mag_tofit),np.squeeze(tdel_tofit),fit_type=0,constrainflux=True,flux_sigma=flux_sigma)

    startmod = import_deflector_params(out + '.dat')

    x_true_fit, y_true_fit, mag_true_fit, t_true_fit, SIE_mod, src_fit = read_write_data(out+'.dat', '',
                                                                                         readonly=True)
    ################## Do fits on mock lenses ##################
    init(path.gravlens_input_path_dump)

    dfilename = path.gravlens_input_path_dump + 'data_in_' + str(dfileind) + '.txt'
    dumppath = path.gravlens_input_path_dump + 'init_' + str(dfileind)
    out = path.gravlens_input_path_dump + 'simout_' + path.cuspfold

    with open(dfilename,'w') as f:
        f.close()

    do_fit(dfilename, dumppath, out, lensmod_infile, path,np.squeeze(xpos_tofit), np.squeeze(ypos_tofit),
           np.squeeze(mag_tofit),np.squeeze(tdel_tofit),fit_type=2,writecosmo=True,write_setgrid=True,startmod=startmod,
           subhalos=subhalos,subprof=subprof,constrainflux=True,flux_sigma=flux_sigma)

    for i in range(1,Nsims+1):
        readfile = out+str(i)+'.dat'
        x_fit,y_fit,mag_fit,t_fit,SIE_mod,src_fit = read_write_data(readfile, '', readonly=True)
        if i>1:
            xfit = np.vstack((xfit,x_fit))
            yfit = np.vstack((yfit,y_fit))
            tfit = np.vstack((tfit,t_fit))
            magfit = np.vstack((magfit,mag_fit))
            SIEmod.append(SIE_mod)
            srcfit = np.vstack((srcfit,np.array(src_fit)))
        else:
            xfit,yfit,magfit,tfit,SIEmod,srcfit = x_fit,y_fit,mag_fit,t_fit,[SIE_mod],src_fit

    shutil.rmtree(path.gravlens_input_path_dump)
    return xfit, yfit, magfit, tfit, SIEmod, srcfit, x_true_fit, y_true_fit, mag_true_fit, t_true_fit

def quick_lensmodel(data_to_fit,constrainflux=True,flux_sigma=.5,tdel_sigma=2,position_simga=0.003):

    baseDIR = os.getenv('HOME') + '/'
    path_to_output = baseDIR + 'data/gravlens_input/temp2/'

    fname = path_to_output+'datafile.txt'
    xpos,ypos,mag,tdel = data_to_fit[0],data_to_fit[1],data_to_fit[2],data_to_fit[3]

    write_data_file(fname, xpos, ypos, mag, tdel, extras=False, fit_type=0, constrainflux=constrainflux,
                    flux_sigma=flux_sigma,position_simga=position_simga,tdel_sigma=tdel_sigma)

    fnamelensmod = path_to_output+'lensmodelin.in'
    dpath = os.getenv('HOME')+'/data/gravlens_input/temp2/'

    with open(fnamelensmod,'w') as f:
        f.write('set omega=0.3\nset lambda=0.7\nset hval=0.7\nset zlens=0.5\nset zsrc=1.5\nset shrcoords=1\nset omitcore=0.001\nset checkparity=0\n')
        f.write('data '+dpath+'datafile.txt\n')
        f.write('set gridflag = 0\nset chimode = 0\nset restart = 3\n\n')
        f.write('setlens 1 1\nalpha 1 0 0 .01 .01 .01 .01 0 0 1\n')
        f.write('1 1 1 1 1 1 1 0 0 0\n')
        f.write('optimize '+dpath+'quicklensmodel')

    run_lensmod(path_to_output + 'lensmodelin.in')

    os.remove(fname)

    xpos, ypos, fr, t, macromodel, src = read_write_data(path_to_output+'quicklensmodel.dat','',readonly=True,read_single=True)

    return xpos,ypos,fr,t




