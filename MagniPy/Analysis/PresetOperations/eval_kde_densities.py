from MagniPy.ABCsampler.Chain import ChainFromSamples

def evaluate_denities(chain_name, which_lenses, error, n_pert, nkde_bins, file_ending, tol = 2000):

    chain = ChainFromSamples(chain_name, which_lens=which_lenses,
                                   error=error, n_pert=n_pert)

    chain.eval_KDE(tol = tol, nkde_bins=nkde_bins, save_to_file=True, file_ending=file_ending)
