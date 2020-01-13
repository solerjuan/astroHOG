# This is a routine to prepare test data for the astroHOG software
# 
# Prepared by Juan D. Soler (soler@mpia.de)

import sys
sys.path.append('../')
from astrohog import *

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy import ndimage

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def makecubes(npix=50):

   cube1=np.zeros([npix,npix,npix])
   cube2=np.zeros([npix,npix,npix])

   # 2-axis
   radius=npix/4.
   mask=create_circular_mask(npix, npix, radius=radius)
   cube1+=np.repeat(mask[:, :, np.newaxis], npix, axis=2)

   # 0-axis
   cube1[:,int(np.round(npix/4.)):int(np.round(3.*npix/4.)),int(np.round(npix/4.)):int(np.round(3.*npix/4.))]=1.0  
 
   # 1-axis
   cube1[int(np.round(4.*npix/16.)):int(np.round(12.*npix/16.)),:,int(np.round(7.*npix/16.)):int(np.round(9.*npix/16.))]=2.0 
   cube1[int(np.round(7.*npix/16.)):int(np.round(9.*npix/16.)),:,int(np.round(4.*npix/16.)):int(np.round(12.*npix/16.))]=2.0 

   plt.imshow(cube1[:,int(npix/2.),:], origin='lower') 
   plt.show()

   hdu = fits.PrimaryHDU(cube1)
   hdu.writeto('cube1.fits')
   #import pdb; pdb.set_trace()

makecubes();


