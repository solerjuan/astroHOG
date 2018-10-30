# HOG comparison of 12CO and 13CO towards Taurus using the Goldsmith et al. data
#
#

import sys
import numpy as np
from astropy.io import fits
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

sys.path.append('/Users/soler/Documents/PYTHON/astroHOG/')
from pyastrohog import *
from rgbtools import *

from astropy.wcs import WCS
from reproject import reproject_interp

import os
import imageio

# -----------------------------------------------------------------------------------------------------------
def alignThreeKpcArm(vmin, vmax, ksz=1, suffix='', MakeIntegratedPlots=False):

   v1=vmin; v2=vmax;
   v1str="%4.1f" % vmin
   v2str="%4.1f" % vmax
   limsv=np.array([v1, v2, v1, v2])
   strksz="%i" % ksz

   # ------------------------------------------------------------------------------------
   dir1='' #'/Users/soler/Dropbox/Work/Geen2017/'
   hdu1=fits.open(dir1+'DHT02_Center_bw_mom.fits')
   rawcube1=hdu1[0].data
   refhdr1=hdu1[0].header.copy()
   cube1=np.transpose(rawcube1,axes=(2,0,1))
   hdu1.close()

   zmin1=int(refhdr1['CRPIX1']+(v1-refhdr1['CRVAL1'])/refhdr1['CDELT1'])
   zmax1=int(refhdr1['CRPIX1']+(v2-refhdr1['CRVAL1'])/refhdr1['CDELT1'])
   velvec1=np.arange(v1,v2,refhdr1['CDELT1'])

   REFhdr=refhdr1.copy()
   REFhdr['NAXIS1']=refhdr1['NAXIS2']
   REFhdr['CTYPE1']=refhdr1['CTYPE2']
   REFhdr['CDELT1']=refhdr1['CDELT2']
   REFhdr['CRPIX1']=refhdr1['CRPIX2']
   REFhdr['CRVAL1']=refhdr1['CRVAL2']

   REFhdr['NAXIS2']=refhdr1['NAXIS3']
   REFhdr['CTYPE2']=refhdr1['CTYPE3']
   REFhdr['CDELT2']=refhdr1['CDELT3']
   REFhdr['CRPIX2']=refhdr1['CRPIX3']
   REFhdr['CRVAL2']=refhdr1['CRVAL3'] 
  
   del REFhdr['NAXIS3']
   del REFhdr['CTYPE3']
   del REFhdr['CDELT3']
   del REFhdr['CRPIX3']
   del REFhdr['CRVAL3']
   REFhdr['NAXIS']=2
 
   cube1[np.isnan(cube1)]=0. 
   plt.subplot(projection=WCS(REFhdr))
   plt.imshow(cube1.sum(axis=0),origin='lower')
   plt.show()
 
   # ------------------------------------------------------------------------------------- 
   dir2=''
   hdu2=fits.open(dir2+'redline_b1_for_juan.fits')
   rawcube2=hdu2[0].data
   refhdr2=hdu2[0].header.copy()
   cube2=np.transpose(rawcube2,axes=(2,0,1))
   hdu2.close()

   zmin2=int(refhdr2['CRPIX1']+(v1-refhdr2['CRVAL1'])/refhdr2['CDELT1'])
   zmax2=int(refhdr2['CRPIX1']+(v2-refhdr2['CRVAL1'])/refhdr2['CDELT1'])
   velvec2=np.arange(v1,v2,refhdr2['CDELT1'])
 
   TARhdr=refhdr2.copy()
   TARhdr['NAXIS1']=refhdr2['NAXIS2']
   TARhdr['CTYPE1']=refhdr2['CTYPE2']
   TARhdr['CDELT1']=refhdr2['CDELT2']
   TARhdr['CRPIX1']=refhdr2['CRPIX2']
   TARhdr['CRVAL1']=refhdr2['CRVAL2']

   TARhdr['NAXIS2']=refhdr2['NAXIS3']
   TARhdr['CTYPE2']=refhdr2['CTYPE3']
   TARhdr['CDELT2']=refhdr2['CDELT3']
   TARhdr['CRPIX2']=refhdr2['CRPIX3']
   TARhdr['CRVAL2']=refhdr2['CRVAL3']

   del TARhdr['NAXIS3']
   del TARhdr['CTYPE3']
   del TARhdr['CDELT3']
   del TARhdr['CRPIX3']
   del TARhdr['CRVAL3']
   TARhdr['NAXIS']=2

   cube2[np.isnan(cube2)]=0.
   plt.subplot(projection=WCS(TARhdr))
   plt.imshow(cube2.sum(axis=0),origin='lower')
   plt.show()

   sz2=np.shape(cube2)
   outcube=np.zeros([np.size(velvec1),sz2[1],sz2[2]])

   for i in range(zmin1,zmax1):
      print(velvec1[i-zmin1])
 
      hduTEMP=fits.PrimaryHDU(cube1[i,:,:])
      hduTEMP.header=REFhdr
      array, footprint = reproject_interp(hduTEMP, TARhdr)

      outcube[i-zmin1,:,:]=array

   plt.subplot(projection=WCS(TARhdr))
   plt.imshow(outcube.sum(axis=0), origin='lower', cmap='binary_r')
   plt.grid(color='white', ls='solid')
   plt.xlabel('GLON')
   plt.ylabel('GLAT')
   plt.show()

   hduOUT=fits.PrimaryHDU(outcube)
   OUThdr=TARhdr.copy()
   OUThdr['CTYPE3']=refhdr1['CTYPE1']
   OUThdr['CDELT3']=refhdr1['CDELT1']
   OUThdr['CRPIX3']=0
   OUThdr['CRVAL3']=velvec1[0]
   hduOUT.header=OUThdr
   hduOUT.writeto('hogDHT02_Center_bw_mom.fits', overwrite=True)

   #import pdb; pdb.set_trace()

alignThreeKpcArm(-150.,50.);

