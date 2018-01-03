import os

class GalProperties:

    def __init__(self,objname):

        self.objname=objname

    def get_all(self,img_config=''):

        self.get_props()

        if len(img_config)!=0:
            path_2_ref = os.getenv('HOME')+'/data/lensdata/'+self.objname\
            +'/imgpos_reference_'+img_config+'.txt'
            try:
                self.src = self.get_srcpos(path_2_ref)
            except:
                print 'cannot locate file '+path_2_ref
                self.src = [0,0]
        else:
            self.src=[]

        self.get_props()
        refind=self.get_imginfo(img_config=img_config)

        return self.ks,self.rs,self.RE,self.rhalf,\
               self.shear,self.shearPA,refind,self.src

    def get_props(self):

        if self.objname=='VCC731':

            self.pixscale = 0.00084
            self.rs = 9.674
            self.rhalf = 1.9349
            self.RE = 1.11
            self.ks = 0.0757
            self.shear = 0.05
            self.shearPA = 70

        elif self.objname=='VCC1632':

            self.pixscale = 0.00064
            self.rs = 5.45
            self.rhalf = 1.09
            self.RE = 1.02
            self.ks = 0.1249
            self.shear = 0.05
            self.shearPA = 55

        elif self.objname == 'VCC1903':

            self.pixscale = 0.00064
            self.rs = 5*1.36771187
            self.rhalf = 1.36771187
            self.RE = .85
            self.ks = 0.0827
            self.shear = 0.05
            self.shearPA = 17

        elif self.objname == 'NGC4874':

            self.pixscale = 0.0038
            self.rs = 5*1.8103
            self.rhalf = 1.8103
            self.RE = 1.1623
            self.ks = 0.0819
            self.shear = 0.05
            self.shearPA = 45

        elif self.objname == 'NGC5322':

            self.pixscale = 0.00134
            self.rs = 5*0.7135
            self.rhalf = 0.7135
            self.RE = 0.9149
            self.ks = 0.1617
            self.shear = 0.05
            self.shearPA = 57

        elif self.objname == 'NGC4872':

            self.pixscale = 0.0038
            self.rs = 5*2.14873
            self.rhalf = 2.14873
            self.RE = 0.7800507
            self.ks = 0.0478
            self.shear = 0.05
            self.shearPA = 70

        elif self.objname == 'NGC5557':

            self.pixscale = 0.0021663421251
            self.rs = 5*0.996517378576
            self.rhalf = 0.996517378576
            self.RE = 1.0598
            self.ks = 0.1342
            self.shear = 0.05
            self.shearPA = 23

        elif self.objname == 'NGC1132':

            self.pixscale = 0.0038
            self.rs = 5*1.22458
            self.rhalf = 1.22458
            self.RE = 0.9941
            self.ks = 0.101
            self.shear = 0.05
            self.shearPA = 135

        elif self.objname == 'NGC7626':

            self.pixscale = 0.002166
            self.rs = 5*0.87087
            self.rhalf = 0.87087
            self.RE = 1.233
            self.ks = 0.1681
            self.shear = 0.05
            self.shearPA = 121

        elif self.objname == 'VCC881':

            self.pixscale = 0.00064
            self.rs = 5*5.274
            self.rhalf = 5.274
            self.RE = 0.86899
            self.ks = 0.0202
            self.shear = 0.05
            self.shearPA = 85

        elif self.objname == 'NGC1272':

            self.pixscale = 0.00298157789368
            self.rs = 5*1.23437
            self.rhalf = 1.23437
            self.RE = 1.4006
            self.ks = 0.1348
            self.shear = 0.05
            self.shearPA = 23

        elif self.objname == 'VCC2000':

            self.pixscale = 0.00064
            self.rs = 5*0.133825737122
            self.rhalf = 0.133825737122
            self.RE = 0.605565856591
            self.ks = 0.5758
            self.shear = 0.05
            self.shearPA = 75

        else:
            self.pixscale = None
            self.rs = None
            self.rhalf = None
            self.RE = None
            self.ks = None
            self.shear = None
            self.shearPA = None


    def get_srcpos(self,path2file):

        with open(path2file,'r') as f:
            lines = f.readlines()
        srcline = lines[0].split(' ')
        srcline = filter(None, srcline)
        return [float(srcline[0]),float(srcline[1])]

    def get_imginfo(self,img_config=False):

        # default
        self.refindex = 1

        if img_config=='fold':
            self.refindex = 1

        else:
            if self.objname=='VCC731':
                if img_config=='cusp':
                    self.refindex = 1
                elif img_config=='cusp2':
                    self.refindex = 2
            elif self.objname=='VCC1632':
                if img_config=='cusp':
                    self.refindex = 1
            elif self.objname=='VCC1903':
                if img_config=='cusp':
                    self.refindex = 2
            elif self.objname=='NGC4872':
                if img_config=='cusp':
                    self.refindex=2
            elif self.objname=='NGC4874':
                if img_config=='cusp':
                    self.refindex = 1
            elif self.objname=='VCC1903':
                if img_config=='cusp':
                    self.refindex=2
            elif self.objname=='NGC5557':
                if img_config=='cusp':
                    self.refindex = 1
            elif self.objname=='NGC1132':
                if img_config=='cusp':
                    self.refindex = 1
            elif self.objname=='VCC881':
                if img_config == 'cusp':
                    self.refindex = 1
            elif self.objname=='VCC2000':
                if img_config == 'cusp':
                    self.refindex = 2
        return self.refindex









