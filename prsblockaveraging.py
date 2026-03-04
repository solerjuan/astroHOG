#
#
#

import sys
import numpy as np
from astropy.io import fits
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle

from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.stats import gaussian_fwhm_to_sigma
from reproject import reproject_interp
import pycircstat as circ

import os
import imageio
from scipy import stats
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

#basedir="/home/soler/ongoing_work/"
#sys.path.append(basedir+'PYTHON/astroHOG/')
from statests import * 
from astrohog2d1v import *
#from rgbtools import *

# --------------------------------------------------------------------------
def prsblocks2D(corrslice, nbx=7, nby=7, weight=1., refmap=None, mask1=None, mask2=None):

   vblocks=np.zeros_like(corrslice)
   vdblocks=np.zeros_like(corrslice)
   vmaxblocks=np.zeros_like(corrslice)
   ngoodblocks=np.zeros_like(corrslice)
   footprint=np.zeros_like(corrslice)

   sz=np.shape(corrslice)
   xvec=np.floor(np.arange(sz[0])/(sz[0]/nbx))
   yvec=np.floor(np.arange(sz[1])/(sz[1]/nbx))
   xx, yy = np.meshgrid(xvec.astype(int), yvec.astype(int))

   for i in range(0,nbx):
      for k in range(0, nby):
         good=np.logical_and(xx==i, yy==k).nonzero()
         footprint[good]=1.0

         phi=np.ravel(corrslice[good])
         wghts=weight*np.ones_like(phi[np.isfinite(phi).nonzero()])
 
         ngoodblocks[good]=np.size(np.isfinite(phi).nonzero())

         if (np.size(np.isfinite(phi).nonzero()) > 0):
            output=HOG_PRS(2.*phi[np.isfinite(phi).nonzero()], weights=wghts)
            vblocks[good]=output['Zx'] 
            output=HOG_PRS(phi[np.isfinite(phi).nonzero()], weights=wghts)
            vdblocks[good]=output['Zx']
            output=HOG_PRS(0.*phi[np.isfinite(phi).nonzero()], weights=wghts) 
            vmaxblocks[good]=output['Zx'] 
         else:
            vblocks[good]=np.nan
            vdblocks[good]=np.nan
            vmaxblocks[good]=np.nan
   output={'V': vblocks, 'Vd': vdblocks, 'Vmax': vmaxblocks}

   return output

