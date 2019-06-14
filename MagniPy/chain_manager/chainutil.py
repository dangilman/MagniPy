class KeywordParse(object):

    labels_short = ['srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mhm', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro', 'satx', 'saty', 'satrein']
    labels = [r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$', r'$m_{\rm{hm}}$',
                       r'$\theta_E$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$',
                       r'$G2_{\theta_E}$', r'$G2_{x}$', r'$G2_{y}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:
            for i, label in enumerate(self.labels_short):
                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    column_inds.append(i)
                    break
        return new_labs, pranges, column_inds
