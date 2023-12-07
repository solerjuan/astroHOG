# This is an example of the comparison of images using the HOG technique
#
# Prepared by Juan D. Soler (soler@mpia.de)

import sys
sys.path.append('../')
from astrohog2d import *
from statests import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy import ndimage

# Load the images that you want to compare
# astroHOG assumes that the cubes are spatially aligned and are reprojected into the same grid

hdul = fits.open('../data/testimage1.fits')
image1=hdul[0].data
hdul.close()
hdul = fits.open('../data/testimage2.fits')
image2=hdul[0].data
hdul.close()

# In case the images are in png format
#image1 = scipy.ndimage.imread('../data/image.001.png', flatten=True)
#image2 = scipy.ndimage.imread('../data/image.002.png', flatten=True)

fig = plt.figure(figsize=(12.0, 6.0))
plt.rc('font', size=10)
ax1=plt.subplot(121)
ax1.imshow(image1, origin='lower', cmap='Greys_r', interpolation='none')
ax1.set_title('Image 1')
ax2=plt.subplot(122)
ax2.imshow(image2, origin='lower', cmap='Greys_r', interpolation='none')
ax2.set_title('Image 2')
plt.show()

# Here you select the size of your derivative kernel in pixels
ksz=9

# Here I define the masks for both images
# For the sake of simplicity, I'm only masking the edges
sz1=np.shape(image1)
mask1=1.+0.*image1
mask1[0:ksz,:]=0.
mask1[sz1[0]-1-ksz:sz1[0]-1,:]=0.
mask1[:,0:ksz]=0.
mask1[:,sz1[1]-1-ksz:sz1[1]-1]=0.
sz2=np.shape(image2)
mask2=1.+0.*image2
mask2[0:ksz,:]=0.
mask2[sz2[0]-1-ksz:sz2[0]-1,:]=0.
mask2[:,0:ksz]=0.
mask2[:,sz2[1]-1-ksz:sz2[1]-1]=0.

# Calculate the relative orientation angles using the tools in the astroHOG package
circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_imaLITE(image1, image2, ksz=ksz, mask1=mask1, mask2=mask2)

# Print the correlation statistics obtained by astrohog
print('Mean resultant vector (r)        ', circstats['RVL'])
print('Rayleigh statistic (Z)           ', circstats['Z'])
print('Projected Rayleigh statistic (V) ', circstats['V'])
print('Pearson correlation coefficient  ', circstats['pearsonr'])
print('Cross correlation                ', circstats['crosscor'])
print('Number of gradient pairs         ', circstats['ngood'])


# Calculate the relative orientation angles and uncertainties using standard deviation values
circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_ima(image1, image2, ksz=ksz, mask1=mask1, mask2=mask2, s_ima1=0.1*np.nanmin(image1), s_ima2=0.1*np.nanmin(image2), nruns=3)

# Print the correlation statistics obtained by astrohog
print('Mean resultant vector (r)        ', circstats['RVL'], '+/-', circstats['s_RVL'])
print('Rayleigh statistic (Z)           ', circstats['Z'], '+/-', circstats['s_Z'])
print('Projected Rayleigh statistic (V) ', circstats['V'], '+/-', circstats['s_V'])
print('Number of gradient pairs         ', circstats['ngood'])

# For the sake of illustration, here we calculate the gradients that underlie astroHOG
dI1dx=ndimage.filters.gaussian_filter(smoothframe1, [ksz, ksz], order=[0,1], mode='nearest')
dI1dy=ndimage.filters.gaussian_filter(smoothframe1, [ksz, ksz], order=[1,0], mode='nearest')
normgrad1=np.sqrt(dI1dx**2+dI1dy**2)
udI1dx=dI1dx/normgrad1
udI1dy=dI1dy/normgrad1
psi1=np.arctan2(dI1dy,dI1dx)

dI2dx=ndimage.filters.gaussian_filter(smoothframe2, [ksz, ksz], order=[0,1], mode='nearest')
dI2dy=ndimage.filters.gaussian_filter(smoothframe2, [ksz, ksz], order=[1,0], mode='nearest')
normgrad2=np.sqrt(dI2dx**2+dI2dy**2)
udI2dx=dI2dx/normgrad2
udI2dy=dI2dy/normgrad2
psi2=np.arctan2(dI2dy,dI2dx)

# This is just setting up the vector representation
pitch=50
sz=np.shape(image1)
X, Y = np.meshgrid(np.arange(0, sz[1]-1, pitch), np.arange(0, sz[0]-1, pitch))
ux1=dI1dx[Y,X]
uy1=dI1dy[Y,X]
ux2=dI2dx[Y,X]
uy2=dI2dy[Y,X]
ux1=np.cos(psi1[Y,X])
uy1=np.sin(psi1[Y,X])
ux2=np.cos(psi2[Y,X])
uy2=np.sin(psi2[Y,X])

fig = plt.figure(figsize=(12.0, 6.0))
plt.rc('font', size=10)
ax1=plt.subplot(121)
ax1.imshow(normgrad1, origin='lower', cmap='Greys_r', interpolation='none')
ax1.set_title('Gradient of Image 1')
ax2=plt.subplot(122)
ax2.imshow(normgrad2, origin='lower', cmap='Greys_r', interpolation='none')
ax2.set_title('Gradient of Image 2')
plt.show()

fig = plt.figure(figsize=(12.0, 6.0))
plt.rc('font', size=10)
ax1=plt.subplot(121)
ax1.imshow(normgrad1, origin='lower', cmap='Greys_r', interpolation='none')
arrows1=ax1.quiver(X, Y, ux1, uy1, units='width', color='cyan', pivot='middle', width=0.005, headwidth=2, headlength=4)
ax1.set_title('Image 1 gradient magnitude and direction')
ax2=plt.subplot(122)
ax2.imshow(normgrad2, origin='lower', cmap='Greys_r', interpolation='none')
arrows2=ax2.quiver(X, Y, ux2, uy2, units='width', color='orange',  pivot='middle', width=0.005, headwidth=2, headlength=4)
ax2.set_title('Image 2 gradient magnitude and direction')
plt.show()

fig, ax = plt.subplots(figsize=(9., 6.))
arrows1=ax.quiver(X, Y, ux1, uy1, units='width', color='cyan',   pivot='middle',  width=0.005, headwidth=2, headlength=4)
arrows2=ax.quiver(X, Y, ux2, uy2, units='width', color='orange',  pivot='middle', width=0.005, headwidth=2, headlength=4)
plt.show()

fig, ax = plt.subplots(1,1, figsize=(9., 6.))
im=plt.imshow(np.abs(corrframe)*180.0/np.pi, origin='lower', cmap='spring', interpolation='none')
cb1=plt.colorbar(im,fraction=0.046, pad=0.04)
cb1.set_label(r'$|\phi|$ [deg]')
ax.set_title('Relative orientation between gradients')
plt.show()




