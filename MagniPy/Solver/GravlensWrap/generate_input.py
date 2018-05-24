import numpy as np
from MagniPy.util import *
from gravlens_to_kwargs import *

class GravlensInput:

    def __init__(self,filename='',zlens=float,zsrc=float,pos_sigma=None,flux_sigma=None,tdelay_sigma=None,
                 identifier='',dataindex=1,paths=classmethod,cosmology=None):

        self.filename = filename
        self.outfile_path = paths.gravlens_input_path_dump
        self.systems = []
        self.Nsystems = 0
        if pos_sigma is not None or flux_sigma is not None or tdelay_sigma is not None:
            self.xpos_sigma,self.ypos_sigma,self.flux_sigma,self.tdelay_sigma = \
                pos_sigma[0],pos_sigma[1],flux_sigma,tdelay_sigma
        self.cosmology = cosmology
        self.identifier = identifier
        self.dataindex = dataindex

    def add_lens_system(self,system=classmethod):

        self.systems.append(system)
        self.Nsystems += 1

    def write_all(self,data,zlens=float,zsrc=float,srcx=None,srcy=None,opt_routine=None):

        extra_commands = {}

        self._add_header(zlens=zlens, zsrc=zsrc, opt_routine=opt_routine)

        if data is not None:

            self._add_dfile(self.outfile_path+self.filename+'_data'+str(self.dataindex)+'.txt')

            self._write_dfile(self.outfile_path+self.filename+'_data'+str(self.dataindex)+'.txt',data)

        self._write_header()

        outputfile_name = []

        for n,system in enumerate(self.systems):

            if srcx is None and srcy is None:
                outputfile_name.append(self.outfile_path + self.identifier + str(n)+'.dat')
            else:
                outputfile_name.append(self.outfile_path + self.identifier + str(n) + '.txt')


            if self.header.routine == 'randomize':

                extra_commands = {}

                extra_commands['randomize'] = ['randomize 10 ' + self.outfile_path + self.filename + '_rand' + str(self.dataindex) + '\n']

                ranges = ''

                for object in system.single_models:

                    if object.deflector.tovary and object.deflector.profname=='SPEMD':

                        for i,flag in enumerate(object.deflector.varyflags):

                            if float(flag)==1:

                                if i==0:

                                    ranges+='0 1.2\n'

                                elif i==1:

                                    ranges+='-.025,.025\n'

                                elif i == 2:

                                    ranges += '-.025,.025\n'

                                elif i == 3:

                                    ranges += '-.1,.1\n'

                                elif i == 4:

                                    ranges += '-.1,.1\n'

                                elif i == 5:

                                    ranges += '-.1,.1\n'

                                elif i == 6:

                                    ranges += '-.1,.1\n'

                        extra_commands['randomize'] += [ranges]

                        extra_commands['randomize'] += ['set gridflag = 1\nset chimode = 1\nset restart = 2\n',
                            'setlens ' + self.outfile_path + self.filename + '_rand' +
                            str(self.dataindex) + '.start\n',
                            'optimize ' + self.outfile_path + self.identifier + str(0)]

                    break

            self._write_lensmodel(full_lensmodel=system,extra_commands=extra_commands,
                                  outfile=self.outfile_path+self.identifier+str(n),srcx=srcx,srcy=srcy)

        return outputfile_name

    def _add_dfile(self,dfilename=''):

        self.header.inputstring += 'data ' + dfilename+ '\n'

    def _write_dfile(self,dfilename,data):

        with open(dfilename,'w') as f:

            f.write(str(1) + '\n')
            f.write('0.000000e+00 0.000000e+00 40.000000e-03\n')
            f.write(str(0.0) + ' ' + str(10000) + '\n')
            f.write(str(0.0) + ' ' + str(10000) + '\n')
            f.write(str(0.0) + ' ' + str(10000) + '\n')
            f.write(str(1) + '\n' + str(4) + '\n')

            nimgs = int(len(self.xpos_sigma))
            xpos,ypos,mag,tdel = data[0],data[1],data[2],data[3]
            maxflux = np.max(mag)

            for i in range(0, nimgs):

                f.write(str(xpos[i]) + ' ' + str(ypos[i]) + ' ' + str(
                        float(mag[i]) * maxflux ** -1) + ' ' + str(self.xpos_sigma[i]) + ' ' + str(
                        self.flux_sigma[i] * float(mag[i]) * maxflux ** -1) + ' ' + str(tdel[i]) + ' ' + str(self.tdelay_sigma[i]) + '\n')

    def _add_header(self,zlens, zsrc, opt_routine):

        self.header = Header(zlens=zlens,zsrc=zsrc,omega_M=self.cosmology.cosmo.Om0,hval=self.cosmology.h)
        self.header.opt_routine(opt_routine)

    def _write_header(self):

        with open(self.outfile_path+self.filename+'.txt','w') as f:
            f.write(str(self.header.inputstring))
        f.close()

    def _write_lensmodel(self,full_lensmodel = classmethod, extra_commands = None, outfile = '',srcx=None,srcy=None):

        with open(self.outfile_path+self.filename+'.txt','a') as f:
            if len(extra_commands) is not 0:
                if 'randomize' in extra_commands:
                    f.write(full_lensmodel.write(outfile=outfile, srcx=srcx, srcy=srcy))
                    for arg in extra_commands['randomize']:
                        f.write(str(arg))

            else:
                f.write(full_lensmodel.write(outfile=outfile, srcx=srcx, srcy=srcy))

class FullModel:

    def __init__(self,multiplane=True):

        self.single_models = []
        self.Ncomponents = 0
        self.multiplane = multiplane
        self.front_space = '   '

    def populate(self,newmod = classmethod):

        self.single_models.append(newmod)
        self.Ncomponents += 1

    def write(self,outfile,srcx=None,srcy=None):

        tovary_any = False

        return_string = 'setlens '+str(self.Ncomponents)+' 1'
        vary_string = ''

        if self.multiplane:
            return_string += ' 1\n'
        else:
            return_string += '\n'

        for model in self.single_models:

            return_string += self.front_space+model._get_model(multiplane=self.multiplane)+'\n'

            vary_string += model._get_tovary()+ '\n'

            if model.tovary:
                tovary_any = True

        return_string += vary_string

        if srcx is not None and srcy is not None:
            return_string += 'findimg ' + str(srcx) + ' ' + str(srcy) + ' ' + str(outfile) + '.txt\n'

        else:
            return_string += 'optimize '+str(outfile)+'\n'

        return return_string


class SingleModel:

    def __init__(self, lensmodel=classmethod, tovary=False, units='lensmodel', vary_type='optimize'):

        self.name = lensmodel.profname

        self.deflector = lensmodel

        self.tovary = lensmodel.tovary

        self.units = units

    def _get_tovary(self, vary_type='optimize'):

        if self.tovary:
            if vary_type == 'optimize':
                vary_string = ''
                for arg in self.deflector.varyflags:
                    vary_string += str(arg) + ' '

                return vary_string
            else:
                raise ValueError('set up randomize.')
        else:
            return '0 0 0 0 0 0 0 0 0 0'

    def _get_model(self,multiplane=False):

        lensparams = kwargs_to_gravlens(self.deflector)

        if multiplane:
            lensparams += str(self.deflector.redshift)

        return lensparams

class Header:

    def __init__(self, zlens=float, zsrc=float, omega_M=None, hval=None):

        inputstring = ''

        inputstring += 'set zlens = ' + str(zlens) + '\nset zsrc = ' + str(zsrc) + '\n'

        inputstring += 'set omega = '+str(omega_M)+'\nset lambda = '+str(1-omega_M)+'\nset hval = '+str(hval)+\
                       '\nset shrcoords=1\nset omitcore=.001\nset checkparity=0\nset clumpmode = 0\n'

        self.inputstring = inputstring+'\n\n'

    def opt_routine(self,routine):

        if routine == 'basic':
            self.inputstring += 'set gridflag = 0\nset chimode = 0\n\n'
            self.routine = 'basic'
        elif routine == 'full':
            self.inputstring += 'set gridflag = 1\nset chimode = 1\n\n'
            self.inputstring += '\n\n'
            self.routine = 'full'
        elif routine == 'randomize' or routine == 'randomize_macro':
            self.inputstring += 'set gridflag = 0\nset chimode = 0\n\n'
            self.routine = 'randomize'
        elif routine is None:
            self.inputstring += 'set gridflag = 1\nset chimode = 1\n\n'
            self.routine = 'solveleq'






