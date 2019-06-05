
import sys
sys.path.append('../')
from astrohog2d import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from astropy.io import fits

image1 = scipy.ndimage.imread('../data/image.001.png', flatten=True)
image2 = scipy.ndimage.imread('../data/image.002.png', flatten=True)

circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_ima(image1, image2, ksz=3.)

print('Mean resultant vector (r)        ', circstats[0])
print('Rayleigh statistic (Z)           ', circstats[1])
print('Projected Rayleigh statistic (V) ', circstats[2])
print('Rayleigh statistic (ii)          ', circstats[5], '+/-', circstats[6])
print('Mean angle                       ', circstats[7])
print('Alignment measure (AM)           ', circstats[8])

hist, bin_edges = np.histogram(corrframe*180.0/np.pi, density=True, range=[-90.,90.], bins=40)
bin_center=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

fig, ax = plt.subplots(2,3,figsize=(18.5, 10.5))
#import pdb; pdb.set_trace()
ax[0,0].imshow(image1, cmap='Greys_r')
ax[0,1].imshow(image2, cmap='Greys_r')
im=ax[0,2].imshow(np.abs(corrframe)*180.0/np.pi, cmap='spring')
cb1=plt.colorbar(im,fraction=0.046, pad=0.04)
cb1.set_label(r'$|\phi|$ [deg]')
ax[1,1].step(bin_center, hist*100, color='red')
ax[1,1].set_ylabel('Histogram density [%]')
ax[1,1].set_xlabel(r'$\phi$ [deg]')
ax[1,1].set_xticks([-90.,-45.,0.,45.,90.])
#plt.tight_layout()
plt.show()

#import pdb; pdb.set_trace()


