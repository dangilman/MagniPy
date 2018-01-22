class Subhalo:

    def __init__(self, subclass=classmethod, trunc=float, xcoord = float, ycoord = float, redshift=float, **sub_kwargs):

        self.subhalo_args = subclass.params(**sub_kwargs)
        self.profname = self.subhalo_args['name']
        self.subhalo_args['rt'] = trunc
        self.subhalo_args['x'],self.subhalo_args['y'] = xcoord,ycoord
        self.subhalo_args['z'] = redshift

