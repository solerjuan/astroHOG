# This is an example of the comparison of data cubes using the HOG technique
#
# Prepared by Juan D. Soler (soler@mpia.de)

import sys
sys.path.append('../')
from astrohog3d import *

import numpy as np
from astropy.io import fits

hdu=fits.open('../data/cube1.fits')
cube1=hdu[0].data 

circstats, corrframe, scube1, scube2 = HOGcorr_cubeLITE(cube1, cube1, ksz=3.0)

import pdb; pdb.set_trace()

