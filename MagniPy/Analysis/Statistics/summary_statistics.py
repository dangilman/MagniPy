import numpy as np

def quadrature_piecewise(model,observed):

    residuals = model - observed

    residuals = np.sqrt(np.sum(residuals**2,axis=1))

    return residuals

def R(model,observed,config,img_order):

    R = []

    if config == 'cusp':
        R_obs = (observed[0] - observed[1] - observed[2])*(observed[0]+observed[1]+observed[2])**-1
        for row in range(0,int(np.shape(model)[0])):
            R.append((model[row,0] - model[row,1] - model[row,2]) * (model[row,1]+model[row,0]+model[row,2])**-1)

    elif config == 'fold':
        R_obs = (observed[0] - observed[1]) * (observed[0] + observed[1]) ** -1
        for row in range(0, int(np.shape(model)[0])):
            model = model[row, img_order]
            R.append((model[row, 0] - model[row, 1]) * (model[row, 1] + model[row, 0]) ** -1)

    else:
        R_obs = (observed[0] - observed[1] + observed[2] - observed[3]) * (observed[0] + observed[1] + observed[2] + observed[3]) ** -1
        for row in range(0, int(np.shape(model)[0])):

            R.append(
                (model[row, 0] - model[row, 1] + model[row, 2] - model[row, 3]) * (model[row, 1] + model[row, 0] + model[row, 2]+ model[row,0]) ** -1)

    return np.absolute(np.absolute(R) - np.absolute(R_obs))


