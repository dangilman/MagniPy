import numpy as np
import matplotlib.pyplot as plt

def gauss_eval(_x, cov_inv):
    return np.exp(-0.5 * np.dot(np.dot(_x, cov_inv), _x))

def NDgaussian(pranges, params_weight, nkde_bins):

    points = []
    ranges = []
    sigmas = []
    centers = []

    params_varied = [str(key) for key in pranges.keys()]

    for i, pi in enumerate(params_varied):
        points.append(np.linspace(pranges[pi][0], pranges[pi][1], nkde_bins))
        ranges.append(pranges[pi])

        if pi in params_weight:
            sigmas.append(params_weight[pi][1])
            centers.append(params_weight[pi][0])
        else:
            sigmas.append(1e+4)
            centers.append(np.mean(pranges[pi]))

    sigmas = np.diag(sigmas)
    inverse_cov = np.linalg.inv(sigmas)
    dimension = len(pranges)

    X = np.meshgrid(*points)
    cc_center = np.vstack([X[i].ravel() - centers[i] for i in range(len(X))]).T

    z = [gauss_eval(coord, inverse_cov) for j, coord in enumerate(cc_center)]
    kernel = np.reshape(z, tuple([nkde_bins] * dimension))

    norm = nkde_bins ** dimension / np.sum(kernel)

    kernel *= norm ** -1

    return kernel
