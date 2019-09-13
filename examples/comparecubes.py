# This is an example of the comparison of data cubes using the HOG technique
#
# Prepared by Juan D. Soler (soler@mpia.de)

import sys
sys.path.append('../')
from astrohog3d import *

import numpy as np

cube1=np.random.rand(64,64,64)
cube2=np.random.rand(64,64,64)

circstats, corrframe, scube1, scube2 = HOGcorr_cubeLITE(cube1, cube2, ksz=3.0)

import pdb; pdb.set_trace()


