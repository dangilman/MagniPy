import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import matplotlib.pyplot as plt

class Lens2045(object):

    ################################# DATA #####################################
    x = np.array([-1.1084, -1.2422, -1.3961, 0.5184])
    y = np.array([0.8065, 0.5582, 0.0161, -0.1999])
    m = np.array([1., 0.578, 0.739, 0.102])
    sigma_x = np.array([0.005]*4)
    sigma_y = np.array([0.005]*4)
    sigma_m = np.zeros_like(sigma_x)

    ######################### One plausible macromodel ################
    # One fairly good solution for 0810
    macromodel_lensmodels = ['SPEMD', 'SHEAR']
    center_x_src, center_y_src = -0.6224951, 0.17329766
    e1_max, e2_max = -0.048829664713156945, 0.06937458302941069
    lens_kwargs = [{'e1': 0.1910944347912104, 'e2': -0.1461214261192949,
                    'gamma': 2.0, 'center_x': 0.018766050610987385,
                    'center_y': -0.08623258234967904, 'theta_E': 1.162323940212036},
                   {'e1': -0.05905015649675369, 'e2': 0.07468764718912181}]
    ###########################################################################################

    ################################# SOURCE PROPERTIES #####################################
    source_size_pc = 8 # source size in parsecs
    source_axis_ratio = 1

    kpc_per_arcsec = 8.03 # @ z_source = 1.5
    # source size in m.a.s.
    source_size = 0.001 * source_size_pc * kpc_per_arcsec ** -1
    # set grid resolution to resolve source size
    grid_res = source_size * 0.5
    grid_res = 0.005
    source_light_gaussian = LightModel(light_model_list=['GAUSSIAN'])

    source_width_x = source_size
    source_width_y = source_size * source_axis_ratio

    source_brightness = 1  # arbitrary normalization
    source_kwargs = [{'amp': source_brightness, 'sigma_x': source_width_x, 'sigma_y': source_width_y,
                      'center_x': center_x_src, 'center_y': center_y_src}]
    ###########################################################################################

def raytrace_image(lens_model_list, kwargs_lens):
    lensmodel = LensModel(lens_model_list)
    betax, betay = lensmodel.ray_shooting(xx.ravel(), yy.ravel(), kwargs=kwargs_lens)

    return betax.reshape(shape0), betay.reshape(shape0)


def evaluate_source_light(betax, betay, source_kwargs):
    shape0 = np.shape(betax)
    light = LENS.source_light_gaussian.surface_brightness(betax.ravel(), betay.ravel(), source_kwargs)
    return light.reshape(shape0)


def plot_full_lens(lensmodel_list, kwargs_lens, kwargs_source):
    betax, betay = raytrace_image(lensmodel_list, kwargs_lens)

    return evaluate_source_light(betax, betay, kwargs_source)

def plot_image(lensmodel_list, kwargs_lens, kwargs_source, image_x, image_y,
               image_size_arsec):

    res = LENS.grid_res  # m.a.s. per pixel
    N = int(range_arcsec * res ** -1)
    x = image_x + np.linspace(-image_size_arsec, image_size_arsec, 2 * N)
    y = image_y + np.linspace(-image_size_arsec, image_size_arsec, 2 * N)
    xx, yy = np.meshgrid(x, y)
    shape0 = xx.shape
    lensmodel = LensModel(lensmodel_list)
    betax, betay = lensmodel.ray_shooting(xx.ravel(), yy.ravel(), kwargs=kwargs_lens)

    image = evaluate_source_light(betax.reshape(shape0), betay.reshape(shape0), kwargs_source)

    # returns the image and the magnification
    deltaPix = image_size_arsec / len(image)
    return image, np.sum(image) * deltaPix ** 2

############ Initialize lens and image sizes ############
LENS = Lens2045()
range_arcsec = 0.55
extent_full = [-range_arcsec, range_arcsec, -range_arcsec, range_arcsec]
range_arcsec_individual = 0.05
extent_individual = [-range_arcsec_individual, range_arcsec_individual,
               -range_arcsec_individual, range_arcsec_individual]

# Set the grid size and grid resolution
res = LENS.grid_res # m.a.s. per pixel
N = int(range_arcsec * res ** -1)
x = y = np.linspace(-range_arcsec, range_arcsec, 2*N)
xx, yy = np.meshgrid(x, y)
shape0 = xx.shape

################ SPECIFY THE LENSMODEL ################
images_x, images_y = LENS.x, LENS.y
model_list, kwargs_lens = LENS.macromodel_lensmodels, LENS.lens_kwargs
kwargs_source = LENS.source_kwargs
####################################################

# Plot the four images individually
if False:
    magnifications = []
    for (xcoord, ycoord) in zip(images_x, images_y):
        image, mag = plot_image(model_list, kwargs_lens, kwargs_source, xcoord, ycoord, 0.05)

        plt.imshow(image, origin='lower', cmap = 'bone',
                   extent=extent_individual)
        magnifications.append(mag)
        plt.show()
    print('fluxes: ', magnifications)
    print('flux ratios: ', np.array(magnifications[1:])*magnifications[0]**-1)

    # Plot the full lens
    full_lens_image = plot_full_lens(model_list, kwargs_lens, kwargs_source)
    plt.imshow(full_lens_image, origin='lower', cmap = 'bone', extent = extent_full)
    plt.show()


srcsizes = [2, 8, 12, 20]
fig = plt.figure(1)
fig.set_size_inches(8,8)
counter = 0
window_size = 0.05
for k, src in enumerate(srcsizes):

    kwargs_source[0]['sigma_x'] = src * 0.001
    kwargs_source[0]['sigma_y'] = src * 0.001
    magnifications = []
    axes = []
    for (xcoord, ycoord) in zip(images_x, images_y):
        index = k + counter + 1
        newax = plt.subplot(4,5,index)
        axes.append(newax)
        image, mag = plot_image(model_list, kwargs_lens, kwargs_source, xcoord, ycoord, window_size)

        newax.imshow(image, origin='lower', cmap='viridis',
                   extent=extent_individual)
        newax.annotate('source size:\n'+str(src)+' pc', xy=(0.55,0.8), color='w', xycoords='axes fraction', fontsize=6)
        magnifications.append(mag)
        counter += 1
        newax.axis('off')
    mags = np.array(magnifications) / np.max(magnifications)
    for j in range(0,4):
        axes[j].annotate('flux: ' + str(np.round(mags[j], 2)), xy=(0.55, 0.7), color='w', xycoords='axes fraction',
                       fontsize=6)
    full_lens_image = plot_full_lens(model_list, kwargs_lens, kwargs_source)
    index = k + counter + 1
    newax = plt.subplot(4, 5, index)
    newax.imshow(full_lens_image, origin='lower', cmap='viridis', extent=extent_full)
    newax.axis('off')
plt.subplots_adjust(left=0, bottom=0.1, right=1.6, top=0.9, wspace=0.4, hspace=0.5)
plt.savefig('2045_sourcesizes.pdf')
plt.show()

# To change the source size, axis ratio, etc. edit the values in the Lens0810 class
