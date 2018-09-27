from astropy.io import fits
import numpy as np

def genppvcubes(nside=64,nchannels=24):

   cube1=np.zeros([nchannels,nside,nside])
   cube2=np.zeros([nchannels,nside,nside])

   for i in range(0,nchannels):
      cube1[i,:,:]=np.random.rand(nside,nside)
      cube2[i,:,:]=np.random.rand(nside,nside)

   hdu1=fits.PrimaryHDU(cube1)
   hdu2=fits.PrimaryHDU(cube2)

   hdul=fits.HDUList([hdu1])
   hdul.writeto('ppv1.fits', clobber=True)
   hdul.close()
   
   hdul=fits.HDUList([hdu2])
   hdul.writeto('ppv2.fits', clobber=True)
   hdul.close()
    
genppvcubes()

