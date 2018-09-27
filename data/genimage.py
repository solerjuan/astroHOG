from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
import numpy as np

# --------------------------------------------------------------------------------------
def genimage(nside=512):

   im1=Gaussian2DKernel(nside/8.,x_size=nside)

   im0=np.random.rand(nside,nside)
   im0[nside/2-nside/4 :nside/2+nside/4 ,nside/2-nside/4 :nside/2+nside/4 ]=2.
   im0[nside/2-nside/8 :nside/2+nside/8 ,nside/2-nside/8 :nside/2+nside/8 ]=3.
   im0[nside/2-nside/16:nside/2+nside/16,nside/2-nside/16:nside/2+nside/16]=4.
   im0[nside/2-nside/32:nside/2+nside/32,nside/2-nside/32:nside/2+nside/32]=5.
   im0[nside/2-nside/64:nside/2+nside/64,nside/2-nside/64:nside/2+nside/64]=6.

   im2=convolve_fft(im0, Gaussian2DKernel(nside/64.), boundary='wrap')

   hdu1=fits.PrimaryHDU(im1)
   hdu2=fits.PrimaryHDU(im2)

   hdul=fits.HDUList([hdu1])
   hdul.writeto('image1.fits', clobber=True)
   hdul.close()
   
   hdul=fits.HDUList([hdu2])
   hdul.writeto('image2.fits', clobber=True)
   hdul.close()
    
genimage()

