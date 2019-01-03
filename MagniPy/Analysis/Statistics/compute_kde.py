from MagniPy.ABCsampler.process import compute_joint_kde
import sys

chain_name = 'CDM_sigma0.02_srcsize0.033'

lens_index = int(sys.argv[1])
error = 2
compute_joint_kde(chain_name, lens_index, 40, error, n_pert = 15)
error = 4
compute_joint_kde(chain_name, lens_index, 40, error, n_pert = 15)
error = 8
compute_joint_kde(chain_name, lens_index, 40, error, n_pert = 15)

