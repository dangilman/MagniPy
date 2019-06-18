class KeywordParse(object):

    labels_short = ['f1','f2','f3','f4','stat','srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mhm', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['re_normed', 'gx_normed', 'gy_normed','eps_normed','epstheta_normed','shear_normed',
                     'sheartheta_normed','gmacro_normed']

    labels = [r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$',r'$S_{\rm{lens}}$',r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$', r'$m_{\rm{hm}}$',
                       r'$\theta_E$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$\langle \theta_E\rangle$', r'$\langle G_x\rangle$', r'$\langle G_y\rangle$',
               r'$\langle \epsilon \rangle$', r'$\langle \theta_{\epsilon} \rangle$',
               r'$\langle \gamma_{\rm{ext}} \rangle$', r'$\langle \theta_{\rm{ext}} \rangle$',
               r'$\langle \gamma_{\rm{macro}} \rangle$']

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
