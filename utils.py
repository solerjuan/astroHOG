#
#
#

from scipy import interpolate
import numpy as np

def interpolate_missing_pixels(image, method='nearest', fill_value=0.):

   h, w = image.shape[:2]
   xx, yy = np.meshgrid(np.arange(w), np.arange(h))
   mask=np.isnan(image)

   known_x = xx[~mask]
   known_y = yy[~mask]
   known_v = image[~mask]
   missing_x = xx[mask]
   missing_y = yy[mask]

   interp_values = interpolate.griddata((known_x, known_y), known_v, (missing_x, missing_y), method=method, fill_value=fill_value)
   interp_image = image.copy()
   interp_image[missing_y, missing_x] = interp_values

   return interp_image

# -------------------------------------------------------------------------------
def fill_missing_pixels(image, method='nearest', fill_value=0.):

   h, w = image.shape[:2]
   xx, yy = np.meshgrid(np.arange(w), np.arange(h))
   mask=np.isnan(image)

   known_x = xx[~mask]
   known_y = yy[~mask]
   known_v = image[~mask]
   missing_x = xx[mask]
   missing_y = yy[mask]

   interp_image = image.copy()
   interp_image[missing_y, missing_x] = fill_value

   return interp_image

# -------------------------------------------------------------------------------
def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset):

   return np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0) + offset

def residuals(p, x, y, z):

   height = p["height"].value
   cen_x = p["centroid_x"].value
   cen_y = p["centroid_y"].value
   sigma_x = p["sigma_x"].value
   sigma_y = p["sigma_y"].value
   offset = p["background"].value

   return (z - height*gaussian2D(x,y, cen_x, cen_y, sigma_x, sigma_y, offset))


# -------------------------------------------------------------------------------
def calc_acorientation(image):

   corr=signal.correlate2d(image, image, boundary='fill', mode='full')

   sz=np.shape(corr)
   xx, yy = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]))

   initial=Parameters()
   initial.add("height",value=np.nanmean(corr))
   initial.add("centroid_x",value=0.5*sz[0])
   initial.add("centroid_y",value=0.5*sz[1])
   initial.add("sigma_x",value=0.25*sz[0])
   initial.add("sigma_y",value=0.25*sz[1])
   initial.add("background",value=0.)

   fit=minimize(residuals, initial, args=(xx, yy, corr))
   output=fit.params.valuesdict()

   sigmas=[output['sigma_x'],output['sigma_y']]

   asymI=np.nanmax(sigmas)/np.nanmin(sigmas)
   alphaI=np.arctan2(output['sigma_y'],output['sigma_x'])

   return asymI, alphaI



