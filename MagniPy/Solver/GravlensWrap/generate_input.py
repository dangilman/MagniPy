import numpy as np

class GravlensInput:

    def __init__(self,filename='',outfile_base ='',zlens=float,zsrc=float,pos_sigma=[],flux_sigma=[],tdelay_sigma=[]):

        self.filename = filename
        self.outfile_base = outfile_base
        self.systems = []
        self.Nsystems = 0
        self.xpos_sigma,self.ypos_sigma,self.flux_sigma,self.tdelay_sigma = \
            pos_sigma[0],pos_sigma[1],flux_sigma,tdelay_sigma

    def add_lens_system(self,system=classmethod):

        self.systems.append(system)
        self.Nsystems += 1

    def _add_dfile(self,dfilename=''):

        self.header.inputstring += 'data ' + dfilename + '.txt\n'

    def write_all(self,data,dfilename='',zlens=float,zsrc=float,pre_commands=None):

        self._write_header(fname=self.filename,zlens=zlens,zsrc=zsrc)

        self._add_dfile(dfilename)

        self._write_dfile(dfilename,data)

        for n,system in enumerate(self.systems):
            self._write_lensmodel(full_lensmodel=system,pre_commands=pre_commands, outfile=self.outfile_base+str(n))

    def _write_dfile(self,dfilename,data):

        with open(self.filename+'_data.txt','w') as f:

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

    def _write_header(self,fname, zlens, zsrc):

        self.filename = fname
        self.header = Header(zlens=zlens,zsrc=zsrc)

        with open(self.filename+'.txt','w') as f:
            f.write(str(self.header.inputstring))
        f.close()

    def _write_lensmodel(self,full_lensmodel = classmethod, pre_commands = None, outfile = ''):

        with open(self.filename+'.txt','a') as f:
            if pre_commands is not None:
                f.write(str(pre_commands))

            f.write(full_lensmodel.write(outfile=outfile))

class FullModel:

    def __init__(self,multiplane=True):

        self.single_models = []
        self.Ncomponents = 0
        self.multiplane = multiplane
        self.front_space = '   '

    def populate(self,newmod = classmethod):

        self.single_models.append(newmod)
        self.Ncomponents += 1

    def write(self,outfile):

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

        if tovary_any:
            return_string += 'optimize '+str(outfile)+'\n'

        return return_string


class SingleModel:

    def __init__(self, lensmodel=classmethod, tovary=False, tovary_args=[], vary_type='optimize'):

        self.name = lensmodel.profname

        self.lensmodel = lensmodel

        self.tovary = tovary

    def _get_model(self,multiplane=False):

        name,lensparams = '',[0]*10

        if self.name == 'NFW':
            name = 'tnfw3'
            lensparams[0] = self.lensmodel.lens_args['ks']
            lensparams[1],lensparams[2] = self.lensmodel.lens_args['x0'],self.lensmodel.lens_args['y0']
            lensparams[7] = self.lensmodel.lens_args['rs']
            lensparams[8] = self.lensmodel.lens_args['rt']*lensparams[7]**-1
            lensparams[9] = 1

        elif self.name == 'SIE':

            name = 'alpha'
            lensparams[0] = self.lensmodel.lens_args['b']
            lensparams[1], lensparams[2] = self.lensmodel.lens_args['x0'], self.lensmodel.lens_args['y0']
            lensparams[3],lensparams[4] = self.lensmodel.lens_args['ellip'], self.lensmodel.lens_args['ellip_angle']
            lensparams[9]=1

        if multiplane:
            lensparams.append(self.lensmodel.lens_args['z'])

        model = name+' '

        for element in lensparams:
            model += str(element)+' '

        return model

    def _get_tovary(self, args=None, vary_type='optimize'):

        if self.tovary:
            if vary_type == 'optimize':
                vary_string = ''
                for arg in args:
                    vary_string += str(arg) + ' '

                return vary_string
            else:
                raise ValueError('set up randomize.')
        else:
            return '0 0 0 0 0 0 0 0 0 0'


class Header:
    def __init__(self, zlens=float, zsrc=float, hval=0.7):
        inputstring = ''
        inputstring += 'set zlens = ' + str(zlens) + '\nset zsrc = ' + str(zsrc) + '\n'
        inputstring += 'set omega = 0.3\nset lambda = 0.7\nset hval = 0.7\nset shrcoords=1\nset omitcore=.001\nset checkparity=0\n'
        self.inputstring = inputstring+'\n\n\n'



