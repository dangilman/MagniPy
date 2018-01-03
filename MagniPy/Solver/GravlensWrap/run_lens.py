import subprocess

import grav_input
import lens_simulations.DM_generator.dm_realization as DM
import lens_simulations.data_visualization.data_handling as data
from lens_simulations.gravlens_wrapper.deflector_params import GalProperties as macro_params
from lens_simulations.lens_routines import lens_mapping
from lens_simulations.lens_routines.lens_profiles import *
from run_lensmodel import *


class run_lenssim:
    def __init__(self, input_models, objname=False, key=False, gensubhalos=False, cuspfold=False,
                 Nrealizations=1,kapmap=False, imgpos_ref=True, srcsize=.0012,
                 position_sigma=float,data_4_cumu=False,filter_spatial=False,filter_spatial_2=False,mindis=3,masscut_high=10**11,
                 masscut_low=0,folder='',between_low=float,between_high=float,**kwargs):

        if 'custom_file_name' in kwargs:
            self.custom_file_name = kwargs['custom_file_name']

        self.paths= directory_paths.Paths(objname=objname, cuspfold=cuspfold, key=key,
                                          baseDIR=kwargs['baseDIR'],data_4_cumu=data_4_cumu)
        paths=self.paths

        self.folder=folder

        self.position_sigma = position_sigma
        self.cuspfold = cuspfold
        self.kapmap = kapmap
        self.info_path = paths.infopath
        self.kapmappath = paths.kapmappath
        if kapmap is not False:
            self.kapmap = self.kapmappath+kapmap
        self.gravinputpath = paths.gravlens_input_path
        self.objname = objname
        self.Nrealizations = Nrealizations
        self.gen_subhalos = gensubhalos
        self.imgpos_ref = imgpos_ref
        self.nonquad_x, self.nonquad_y, self.nonquad_imgnum = '', '', ''
        self.temppath = self.paths.temppath
        self.srcsize=srcsize

        if 'fluxerror' in kwargs:
            self.fluxerr = kwargs['fluxerror']
        else:
            self.fluxerr = 0

        self.temppath = self.paths.temppath

        self.key = key

        if input_models == 'load deflector':

            self.macro_type='luminous_gal'
            self.ks, self.rs, self.RE, self.rhalf, self.s, self.sa, self.refindex, src = macro_params(objname).get_all(
                cuspfold)
            self.total_macromodel_kappa = integrate_profile('nfw', limit=self.RE, rs=self.rs, ks=self.ks)
            self.path_defmapx = self.kapmappath + self.objname + '_xdef.fits'
            self.path_defmapy = self.kapmappath + self.objname + '_ydef.fits'
            self.ks, self.rs, self.RE, self.rhalf, self.s, self.sa, self.refindex, src = macro_params(objname).get_all(
                cuspfold)
            self.lens = lens_mapping.Deflector(path_to_defmapx=self.path_defmapx, path_to_defmapy=self.path_defmapy,
                                               gridsize=80)
            proc = subprocess.Popen(['mkdir', paths.temppath])
            proc.wait()
            self.run_extended(Nrealizations=self.Nrealizations, filter_spatial=filter_spatial, mindis=mindis,
                              masscut_low=masscut_low,masscut_high=masscut_high)

        else:

            if input_models=='SIE':

                self.run_extended_SIE(datatofit=kwargs['data_to_fit'],Nrealizations=self.Nrealizations,
                                      filter_spatial=filter_spatial, filter_spatial_2=filter_spatial_2,mindis=mindis,
                                      masscut_low=masscut_low,masscut_high=masscut_high,between_low=between_low,
                                      between_high=between_high,src_size=srcsize)


        proc = subprocess.Popen(['rm', '-r', paths.temppath])
        proc.wait()

    def execute(self, subhalos=False, Nrealizations=1, srccoords=[], **kwargs):

        refsources = grav_input.write_inputfile(paths=self.paths, subhalos=subhalos, input_models=self.input_models,
                                                Nrealizations=Nrealizations, kapmap=self.kapmap, subprofile=self.subprofile, src_basecoords=srccoords).refsrcs

        grav_input.run_lensmod(self.gravinputpath + 'realization.in')

        return refsources

    def run_extended_SIE(self,datatofit=[],Nrealizations=1,filter_spatial=False,filter_spatial_2=False,mindis=3,
                         masscut_low=0,masscut_high=10**10,between_high=float,between_low=float,flux_sigma=0.2,src_size=False):

        xpos_final, ypos_final, mag_final, tdelay_final, srcs_final = np.zeros((Nrealizations, 4)), np.zeros(
            (Nrealizations, 4)), np.zeros((Nrealizations, 4)), np.zeros((Nrealizations, 4)), np.zeros(
            (Nrealizations, 2))

        xtofit = datatofit[0]
        ytofit = datatofit[1]
        mag_tofit = datatofit[2]
        tdel_tofit = datatofit[3]

        self.realization = DM.DM_realization(Rein=1)
        self.realization.set_sub_params(self.key)


        self.input_models = []
        self.nonquad_lensmodels = []

        self.subprofile = self.realization.profile
        if filter_spatial:
            subhalos = self.realization.draw_subhalos(N=Nrealizations,filter_spatial=True,mindis=mindis,
                                                      masscut_low=masscut_low,masscut_high=masscut_high,
                                                      near_x=np.squeeze(xtofit),near_y=np.squeeze(ytofit), Rmax_z=250)
        elif filter_spatial_2:
            subhalos = self.realization.draw_subhalos(N=Nrealizations, filter_spatial_2=True, mindis=mindis,
                                                      between_high=between_high, between_low=between_low,
                                                      near_x=np.squeeze(xtofit), near_y=np.squeeze(ytofit), Rmax_z=250)
        else:

            subhalos = self.realization.draw_subhalos(N=Nrealizations)

        xfit, yfit, magfit, tfit, SIEfit, srcfit, x_true_fit, y_true_fit, mag_true_fit, t_true_fit = \
            run_chainfit(path=self.paths, xpos_tofit=xtofit, ypos_tofit=ytofit, mag_tofit=mag_tofit,
            tdel_tofit=tdel_tofit, subhalos=subhalos, Nsims=self.Nrealizations, subprof=self.subprofile,
            dfileind=1, task_index=1, flux_sigma=1)

        mag_final = get_SIEmags(xfit, yfit, subhalos, SIEfit, srcfit, src_size=src_size, subprofile=self.subprofile,
                                print_status=True)

        if self.paths.baseDIR == '/u/flashscratch/g/gilmanda'+'/':
            self.write_files(xfit, yfit, mag_final, tfit, srcs_final,
                             custom_path=self.paths.baseDIR + self.folder + str(self.custom_file_name))
        else:
            self.write_files(xfit, yfit, mag_final, tfit, srcs_final,
                             custom_path=self.paths.datapath+self.folder+str(self.custom_file_name))


    def run_extended(self, Nrealizations=1,filter_spatial=False,mindis=3,masscut_low=0,masscut_high = 10**11):

        xpos_final, ypos_final, mag_final, tdelay_final, srcs_final = np.zeros((Nrealizations, 4)), np.zeros(
            (Nrealizations, 4)), np.zeros((Nrealizations, 4)), np.zeros((Nrealizations, 4)), np.zeros((Nrealizations, 2))

        DATA = data.read_data(paths=self.paths)
        xref, yref, self.src_basecoords = DATA.refdatax,DATA.refdatay,DATA.src

        Nreal = Nrealizations
        indslist = np.arange(0, Nrealizations, 1)

        if self.gen_subhalos:
            self.realization = DM.DM_realization(Rein=self.RE, rs=self.rs)
            self.realization.set_sub_params(self.key)
        while True:
            self.input_models = []
            self.nonquad_lensmodels = []
            xarray, yarray, magarray, tdelarray = np.zeros((Nreal, 4)), np.zeros((Nreal, 4)), np.zeros(
                (Nreal, 4)), np.zeros((Nreal, 4))

            if self.gen_subhalos:

                self.subprofile = self.realization.profile
                if filter_spatial:
                    subhalos = self.realization.draw_subhalos(N=Nreal,filter_spatial=True,mindis=mindis,masscut_low=masscut_low,
                                                              masscut_high=masscut_high,near_x=np.squeeze(xref),near_y=np.squeeze(yref))
                else:

                    subhalos = self.realization.draw_subhalos(N=Nreal)

                for i in range(0, len(subhalos)):
                    subs = subhalos[i]
                    kappa_sub = DM.convergence_inRE(subs[:, 0], subs[:, 4], subs[:, 5], self.RE,
                                                    self.realization.sigmacrit)
                    rescale = 1 - kappa_sub * self.total_macromodel_kappa ** -1
                    modstring = 'nfw ' + str(self.ks * rescale) + ' 0 0 0 0 ' + str(self.s) + ' ' + str(
                        self.sa) + ' ' + str(
                        self.rs) + ' 0 0'
                    self.input_models.append(modstring)

            else:
                for i in range(0, Nrealizations):
                    modstring = 'nfw ' + str(self.ks) + ' 0 0 0 0 ' + str(self.s) + ' ' + str(self.sa) + ' ' + str(
                        self.rs) + ' 0 0'
                    self.input_models.append(modstring)
                self.subprofile = False
                subhalos = False

            refsrcs = self.execute(subhalos=subhalos, subprofile=self.subprofile, Nrealizations=Nreal,
                                   srccoords=self.src_basecoords)

            x, y, m, t, srcs, inds_kept, inds_to_repeat, fnameinds, not_kept_inds = self.load_quad_data(indslist,
                                                                                                        xarray, yarray,
                                                                                                        magarray,
                                                                                                        tdelarray, xref,
                                                                                                        yref,
                                                                                                        refsrcs)

            xpos_final[fnameinds, :] = np.take(x, inds_kept, axis=0)
            ypos_final[fnameinds, :] = np.take(y, inds_kept, axis=0)
            tdelay_final[fnameinds, :] = np.take(t, inds_kept, axis=0)
            srcs_final[fnameinds, :] = np.take(refsrcs, inds_kept, axis=0)

            print 'computing magnifications...'

            for i in range(0, len(inds_kept)):
                print 'mag: ',len(inds_kept)-i
                self.lens.at_image_location(xpos_final[fnameinds[i], :], ypos_final[fnameinds[i], :])

                if subhalos is not False:
                    subs = subhalos[inds_kept[i]]

                    #subinds = lens_mapping.filter_subs(xpos_final[fnameinds[i], :], ypos_final[fnameinds[i], :],subs[:,4],subs[:,5])

                    #if subinds is not False:
                    #    subs = subs[subinds, :]

                    self.lens.add_subhalos(subx=subs[:, 4], suby=subs[:, 5], sub_b=subs[:, 3], subtrunc=subs[:, 1],
                                      subcore=subs[:, 2],subprofile=self.subprofile)


                mag_final[fnameinds[i], :] = self.lens.img_mag(src_size=self.srcsize,srcx=refsrcs[inds_kept[i], 0],
                                                               srcy=refsrcs[inds_kept[i], 1])

            indslist = inds_to_repeat

            if len(inds_to_repeat) == 0:
                break
            Nreal = len(inds_to_repeat)

        if self.position_sigma>0:
            xpos_final += np.random.normal(0,0.001*self.position_sigma, size=(np.shape(xpos_final)[0], np.shape(xpos_final)[1]))
            ypos_final += np.random.normal(0,0.001*self.position_sigma, size=(np.shape(xpos_final)[0], np.shape(xpos_final)[1]))

        if self.fluxerr!=0:
            for i in range(0,np.shape(mag_final)[0]):

                for j in range(0,np.shape(mag_final)[1]):
                    error = np.random.normal(0,self.fluxerr*mag_final[i,j])
                    mag_final[i, j] += error

                mag_final[i,:]*=max(mag_final[i,:])**-1

        self.write_files(xpos_final, ypos_final, mag_final, tdelay_final, srcs_final)


    def get_macromodel(self,i,macro_type='luminous_gal',subhalos=[]):

        self.realization = DM.DM_realization(Rein=self.RE, rs=self.rs)
        self.subprofile = self.realization.profile
        subhalos.append(self.realization.draw_subhalos(N=1))
        subs = np.squeeze(subhalos[i])
        if subs.ndim == 1:
            kappa_sub = DM.convergence_inRE(subs[0], subs[4], subs[5], self.RE,
                                            self.realization.sigmacrit)
        else:
            kappa_sub = DM.convergence_inRE(subs[:, 0], subs[:, 4], subs[:, 5], self.RE,
                                            self.realization.sigmacrit)
        rescale = 1 - kappa_sub * self.total_macromodel_kappa ** -1

        if macro_type!='SIE':

            rescale = 1 - kappa_sub * self.total_macromodel_kappa ** -1
            modstring = 'nfw ' + str(self.ks * rescale) + ' 0 0 0 0 ' + str(self.s) + ' ' + str(
                self.sa) + ' ' + str(
                self.rs) + ' 0 0'
        else:
            modstring = 'alpha '+str(self.SIE_b*rescale)+' '+str(self.SIEx)+' '+str(self.SIEy)+' '+str(self.SIEellip)+\
                        ' '+str(self.SIEang)+' '+str(self.SIEshear)+' '+str(self.shearang)+' 0 0 1'


        self.input_models.append(modstring)

    def load_quad_data(self, indslist, xvals, yvals, magarray, tdelvals, xref, yref, srcs):
        repeat, count, kept_inds, not_kept_inds = [], 0, [], []
        kept_srcs = np.zeros([self.Nrealizations, 2])

        fnameinds = []

        for k in range(0, len(indslist)):
            ind = indslist[k]

            x, y, mag, tdel, nonquad = data.read_temp_data(n=1).load(k + 1,temp_path=self.temppath)

            if nonquad == 0:
                xvals[k, :] = x
                yvals[k, :] = y
                magarray[k, :] = mag
                tdelvals[k, :] = tdel
                kept_srcs[k, :] = srcs[k, :]
                kept_inds.append(k)
                fnameinds.append(ind)
            else:
                for i in range(0, len(x)):
                    self.nonquad_x += str(x[i])
                    self.nonquad_y += str(y[i])
                self.nonquad_imgnum += str(nonquad)
                self.nonquad_x += '\n'
                self.nonquad_y += '\n'
                self.nonquad_imgnum += '\n'
                repeat.append(ind)
                not_kept_inds.append(k)

        return xvals, yvals, magarray, tdelvals, kept_srcs, kept_inds, repeat, fnameinds, not_kept_inds

    def write_files(self, x=[], y=[], m=[], t=[], srcpos=[], custom_path=None):

        if custom_path is None:
            fname = self.paths.datapath + self.key +'/'+ self.paths.cuspfold + '_data.txt'
        else:
            fname = custom_path

        with open(fname, 'w') as g:
            for i in range(0, np.shape(x)[0]):
                g.write('4 ' + str(srcpos[0, 0]) + ' ' + str(srcpos[0, 1]))
                for j in range(0, 4):
                    g.write(' ' + str(x[i, j]) + ' ' + str(y[i, j]) + ' ' + str(
                        m[i, j] * max(m[i, :]) ** -1) + ' ' + str(t[i, j]))
                g.write('\n')
        g.close()

def quickGravlens(mod,b,x0,y0,ellip,elliptheta,shear,sheartheta,src=[],p8=0,p9=0,p10=1,gridmode=1,shrcoords=None):
    if shrcoords:
        lines='set shrcoords = 1\n'
    else:
        lines='set shrcoords = 2\n'
    lines+='gridmode '+str(gridmode)+'\nset omitcore = 0.01\nsetlens 1 1\n'
    lines+='   '
    lines+=str(mod)+' '+str(b)+' '+str(x0)+' '+str(y0)+' '+str(ellip)+' '+str(elliptheta)+' '+str(shear)+' '+str(sheartheta)+' '+str(p8)+' '+str(p9)+' '+str(p10)+'\n'
    lines+='0 0 0 0 0 0 0 0 0 0\n'
    lines+='findimg '+str(src[0])+' '+str(src[1])+' '+'../gravlens_input/quickout.txt'

    with open('../quickgrav.in','w') as f:
        f.write(lines)
    f.close()

    run_lensmod('../quickgrav.in')

    #os.remove('../quickgrav.in')

    x,y,mag,t,nimg = load_single(fname='../gravlens_input/quickout.txt')

    os.remove('../gravlens_input/quickout.txt')

    return x,y,mag,t,nimg
