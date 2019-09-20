class KeywordParse2033(object):

    labels_short = ['f1','f2','f3','f4','stat','srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mhm', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['re_normed', 'gx_normed', 'gy_normed','eps_normed','epstheta_normed','shear_normed',
                     'sheartheta_normed','gmacro_normed', 'lens_redshift',
                     'satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1', 'satellite_y_1',
                     'satellite_x_2', 'satellite_y_2']

    labels = [r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$', r'$S_{\rm{lens}}$',r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$', r'$m_{\rm{hm}}$',
                       r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$\langle \theta_E\rangle$', r'$\langle G_x\rangle$', r'$\langle G_y\rangle$',
               r'$\langle \epsilon \rangle$', r'$\langle \theta_{\epsilon} \rangle$',
               r'$\langle \gamma_{\rm{ext}} \rangle$', r'$\langle \theta_{\rm{ext}} \rangle$',
               r'$\langle \gamma_{\rm{macro}} \rangle$', r'$z_d$',
               r'$G2_{\theta_E(1)}$', r'$G2_{\theta_E(2)}$', r'$G2_{x(1)}$', r'$G2_{y(1)}$',
                r'$G2_{x(2)}$', r'$G2_{y(2)}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse2033_mcrelation(object):

    labels_short = ['f1','f2','f3','f4','stat','srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'c0', 'beta', 'zeta','re',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift',
                     'satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1', 'satellite_y_1',
                     'satellite_x_2', 'satellite_y_2']

    labels = [r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$', r'$S_{\rm{lens}}$',r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$', r'$c_0$', r'$\beta$', r'$\zeta$',
                       r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',r'$G2_{\theta_E(1)}$', r'$G2_{\theta_E(2)}$', r'$G2_{x(1)}$', r'$G2_{y(1)}$',
                r'$G2_{x(2)}$', r'$G2_{y(2)}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse_mcrelation(object):
    labels_short = ['f1', 'f2', 'f3', 'f4', 'stat', 'srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'c0', 'beta',
                    'zeta', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift','satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1']

    labels = [r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$', r'$S_{\rm{lens}}$', r'$\sigma_{\rm{src}}$',
              r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
              r'$\alpha$', r'$c_0$', r'$\beta$', r'$\zeta$',
              r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
              r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',
               r'$G2_{\theta_E}$', r'$G2_{\theta_E}$', r'$G2_{x}$', r'$G2_{y}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse(object):
    labels_short = ['f1', 'f2', 'f3', 'f4', 'stat', 'srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mhm', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['re_normed', 'gx_normed', 'gy_normed', 'eps_normed', 'epstheta_normed', 'shear_normed',
                     'sheartheta_normed', 'gmacro_normed', 'lens_redshift',
                     'satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1']

    labels = [r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$', r'$S_{\rm{lens}}$', r'$\sigma_{\rm{src}}$',
              r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
              r'$\alpha$', r'$m_{\rm{hm}}$',
              r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
              r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$\langle \theta_E\rangle$', r'$\langle G_x\rangle$', r'$\langle G_y\rangle$',
               r'$\langle \epsilon \rangle$', r'$\langle \theta_{\epsilon} \rangle$',
               r'$\langle \gamma_{\rm{ext}} \rangle$', r'$\langle \theta_{\rm{ext}} \rangle$',
               r'$\langle \gamma_{\rm{macro}} \rangle$', r'$z_d$',
               r'$G2_{\theta_E}$', r'$G2_{\theta_E}$', r'$G2_{x}$', r'$G2_{y}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse2033_varymlow(object):

    labels_short = ['f1','f2','f3','f4','stat','srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mlow', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift',
                     'satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1', 'satellite_y_1',
                     'satellite_x_2', 'satellite_y_2']

    labels = [r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$', r'$S_{\rm{lens}}$',r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$', r'$\log_{10}\left(m_{\rm{min}}\right)$',
                       r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',r'$G2_{\theta_E(1)}$', r'$G2_{\theta_E(2)}$', r'$G2_{x(1)}$', r'$G2_{y(1)}$',
                r'$G2_{x(2)}$', r'$G2_{y(2)}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse_varymlow(object):
    labels_short = ['f1', 'f2', 'f3', 'f4', 'stat', 'srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 'mlow', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift','satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1']

    labels = [r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$', r'$S_{\rm{lens}}$', r'$\sigma_{\rm{src}}$',
              r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
              r'$\alpha$', r'$\log_{10}\left(m_{\rm{min}}\right)$',
              r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
              r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',
               r'$G2_{\theta_E}$', r'$G2_{\theta_E}$', r'$G2_{x}$', r'$G2_{y}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse2033_CDM(object):

    labels_short = ['f1','f2','f3','f4','stat','srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift',
                     'satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1', 'satellite_y_1',
                     'satellite_x_2', 'satellite_y_2']

    labels = [r'$f_1$',r'$f_2$',r'$f_3$',r'$f_4$', r'$S_{\rm{lens}}$',r'$\sigma_{\rm{src}}$', r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
                       r'$\alpha$',
                       r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
                       r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',r'$G2_{\theta_E(1)}$', r'$G2_{\theta_E(2)}$', r'$G2_{x(1)}$', r'$G2_{y(1)}$',
                r'$G2_{x(2)}$', r'$G2_{y(2)}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds

class KeywordParse_CDM(object):
    labels_short = ['f1', 'f2', 'f3', 'f4', 'stat', 'srcsize', 'sigmasub', 'deltalos', 'mparent', 'alpha', 're',
                    'gx', 'gy', 'eps',
                    'epstheta', 'shear', 'sheartheta', 'gmacro']
    labels_short += ['lens_redshift','satellite_theta_E_1', 'satellite_theta_E_2', 'satellite_x_1']

    labels = [r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$', r'$S_{\rm{lens}}$', r'$\sigma_{\rm{src}}$',
              r'$\Sigma_{\rm{sub}}$', r'$\delta_{\rm{los}}$', r'$\log M_{\rm{halo}}$',
              r'$\alpha$', r'$b_{\rm{macro}}$', r'$G1_x$', r'$G1_y$', r'$\epsilon$', r'$\theta_{\epsilon}$',
              r'$\gamma_{\rm{ext}}$', r'$\theta_{\rm{ext}}$', r'$\gamma_{\rm{macro}}$']
    labels += [r'$z_d$',
               r'$G2_{\theta_E}$', r'$G2_{\theta_E}$', r'$G2_{x}$', r'$G2_{y}$']

    def __call__(self, shorthand_labels, ranges):

        new_labs = []
        column_inds = []
        pranges = []

        for lab in shorthand_labels:

            for i, label in enumerate(self.labels_short):

                if lab == label:
                    new_labs.append(self.labels[i])
                    pranges.append(ranges[i])
                    idx = self.labels_short.index(label)
                    column_inds.append(idx)
                    break


        return new_labs, pranges, column_inds
