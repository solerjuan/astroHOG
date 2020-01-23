# This is an example of the comparison of data cubes using the HOG technique
#
# Prepared by Juan D. Soler (soler@mpia.de)

import sys
sys.path.append('../')
from astrohog3d import *

import numpy as np
from astropy.io import fits

hdul = fits.open('/Users/soler/Documents/Data/SILCC/fh.fits')
cube1=hdul[0].data
hdul.close()
hdul = fits.open('/Users/soler/Documents/Data/SILCC/fco.fits')
cube2=hdul[0].data
hdul.close()

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(cube1.sum(axis=0), origin='lower', cmap='magma')
ax[0,0].set_xlabel('2')
ax[0,0].set_ylabel('1')
ax[0,1].imshow(cube1.sum(axis=1), origin='lower', cmap='magma')
ax[0,1].set_xlabel('2')
ax[0,1].set_ylabel('0')
ax[0,2].imshow(cube1.sum(axis=2), origin='lower', cmap='magma')
ax[0,2].set_xlabel('1')
ax[0,2].set_ylabel('0')
ax[1,0].imshow(cube2.sum(axis=0), origin='lower', cmap='magma')
ax[1,0].set_xlabel('2')
ax[1,0].set_ylabel('1')
ax[1,1].imshow(cube2.sum(axis=1), origin='lower', cmap='magma')
ax[1,1].set_xlabel('2')
ax[1,1].set_ylabel('0')
ax[1,2].imshow(cube2.sum(axis=2), origin='lower', cmap='magma')
ax[1,2].set_xlabel('1')
ax[1,2].set_ylabel('0')
plt.show()

circstats, corrframe, scube1, scube2 = HOGcorr_cubeLITE(cube1, cube2, ksz=5.0, weightbygradnorm=True)

print(circstats['meanphi']*180/np.pi)

print(circstats['xi'])


