#
#
import sys
sys.path.append('../')
from astrohog2d import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from astropy.io import fits

nx=100; ny=200
image1=np.random.rand(nx,ny)
image2=np.random.rand(nx,ny)

weights=1.0

fig, ax = plt.subplots(2,figsize=(8.0, 5.0))
ax[0].imshow(image1, cmap='Greys_r')
ax[1].imshow(image2, cmap='Greys_r')
plt.show()

circstat, corrframe, smooth1, smooth2 = HOGcorr_imaLITE(image1, image2, ksz=3.0, weights=weights)
print('V                ', circstat['V'])

circstat0, corrframe0, smooth1_0, smooth2_0 = HOGcorr_imaLITE(image1, image1, ksz=3.0, weights=weights)
print('Theoretical Vmax ', np.sqrt(2.*weights*nx*ny))
print('Empirical Vmax:  ', circstat0['V'])

print('V/Vmax:', 100.*circstat['V']/circstat0['V'])

print(np.sum(np.cos(2.*corrframe)))
import pdb; pdb.set_trace()


