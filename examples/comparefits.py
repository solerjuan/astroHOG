
import sys
sys.path.append('../pyastrohog/')
from astrohog2d import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from astropy.io import fits

hdul=fits.open('../data/image1.fits')
image1=hdul[0].data
hdul.close()
hdul=fits.open('../data/image2.fits')
image2=hdul[0].data
hdul.close()

circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_frame(image1, image2)

print('Mean resultant vector (r)        ', circstats[0])
print('Rayleigh statistic (Z)           ', circstats[1])
print('Projected Rayleigh statistic (V) ', circstats[2])
print('Rayleigh statistic (ii)          ', circstats[5], '+/-', circstats[6])
print('Mean angle                       ', circstats[7])
print('Alignment measure (AM)           ', circstats[8])

hist, bin_edges = np.histogram(corrframe*180.0/np.pi, density=True, range=[-90.,90.], bins=40)
bin_center=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

fig=plt.figure()
ax1=plt.subplot(221)
plt.imshow(image1, cmap='bone', origin='lower')
ax1=plt.subplot(222)
plt.imshow(image2, cmap='copper', origin='lower')
ax1=plt.subplot(223)
im=plt.imshow(np.abs(corrframe)*180.0/np.pi, cmap='spring', origin='lower')
cb1=plt.colorbar(im) #,fraction=0.046, pad=0.04)
cb1.set_label(r'$|\phi|$ [deg]')
ax1=plt.subplot(224)
plt.step(bin_center, hist*100, color='red')
plt.ylabel('Histogram density [%]')
plt.xlabel(r'$\phi$ [deg]')
plt.xticks([-90.,-45.,0.,45.,90.])
plt.tight_layout()
plt.show()

#import pdb; pdb.set_trace()


