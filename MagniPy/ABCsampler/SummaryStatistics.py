import numpy as np

def quadrature_piecewise(model,observed):

    residuals = model - observed

    residuals = np.sqrt(np.sum(residuals**2,axis=1))

    return residuals


