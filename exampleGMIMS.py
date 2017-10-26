# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler
#
# Example of HOG correlation analysis of the Global Magneto-Ionic Medium Survey
#
# 26OCT2017

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

sys.path.append('/Users/jsoler/Documents/astrohog/')
from astrohog import *

from astropy.wcs import WCS
from reproject import reproject_interp

def astroHOGexampleGMIMS(frame, vmin, vmax, ksz=1):
# Calculates the HOG correlation between the GMIMS data and HI? 
# INPUTS
# frame - Currently not doing anything 
# vmin  - Minimum RM  
# vmax  - Maximum RM
# ksz   - Size of the derivative kernel
# 
#
#

   fstr="%4.2f" % frame

# ============================================================================================================
# Reading input data
# ============================================================================================================

# -------------------------------------------------------------------------------------------------------------
# Reading RM cube
# -------------------------------------------------------------------------------------------------------------
   dir='3C196/'
   hdu1=fits.open(dir+'3C196_LOFAR_RMcube_10.fits')
   RMcube=hdu1[0].data
   v1=vmin*1000.
   v2=vmax*1000.
   v1str="%4.1f" % vmin
   v2str="%4.1f" % vmax
   limsv=np.array([v1, v2, v1, v2])

   sz1=np.shape(hdu1[0].data)
   CTYPE3=hdu1[0].header['CTYPE3']
   CDELT3=hdu1[0].header['CDELT3']
   CRVAL3=hdu1[0].header['CRVAL3']
   CRPIX3=hdu1[0].header['CRPIX3']
   zmin1=0        #int(CRPIX3+(v1-CRVAL3)/CDELT3)
   zmax1=sz1[0]-1  #int(CRPIX3+(v2-CRVAL3)/CDELT3)
   velvec1=hdu1[0].header['CRVAL3']+np.arange(sz1[0])*hdu1[0].header['CDELT3']   #np.arange(v1,v2,CDELT3)/1000.

   refhdr=hdu1[0].header.copy()
   NAXIS3=refhdr['NAXIS3']
   del refhdr['NAXIS3']
   del refhdr['CTYPE3']
   del refhdr['CRVAL3']
   del refhdr['CRPIX3']
   del refhdr['CDELT3']
   del refhdr['CUNIT3']
   refhdr['NAXIS']=2

# -------------------------------------------------------------------------------------------------------------
# Reading other files
# -------------------------------------------------------------------------------------------------------------
   hdu3=fits.open(dir+'3C196fwhm30_Qmap.fits')
   hdu4=fits.open(dir+'3C196fwhm30_Umap.fits')

   Qmap=hdu3[0].data
   Umap=hdu4[0].data
   psi=0.5*np.arctan2(-Umap, Qmap)
   ex=np.sin(psi)
   ey=np.cos(psi)

# ==========================================================================================================
# Pixel size  
# =========================================================================================================
   res=40. #arcsec
   pxsz=np.abs(refhdr1['CDELT1'])*60.*60. #arcsec
   pxksz=np.int(np.round(ksz/pxsz))

# ==========================================================================================================
# Compute mask
# ==========================================================================================================
   sz1=np.shape(galRMcube)
   #x=np.sort(galRMcube.ravel())
   #minrm=x[int(0.2*np.size(x))]
   minrm=np.std(galRMcube[0:5,:,:])
   mask1=np.zeros(sz1)
   mask1[(galRMcube > minrm).nonzero()]=1
   mask1[:,0:pxksz,:]=0.; mask1[:,sz1[1]-pxksz-1:sz1[1]-1,:]=0.
   mask1[:,:,0:pxksz]=0.; mask1[:,:,sz1[2]-pxksz-1:sz1[2]-1]=0.

   mask2=np.zeros(sz2)+1.
   #mask2[:,ymin:ymax,:]=1
   #mask2[(hdu2[0].data < 0.0).nonzero()]=0
   mask2[:,0:pxksz,:]=0.; mask2[:,sz2[1]-pxksz-1:sz2[1]-1,:]=0.
   mask2[:,:,0:pxksz]=0.; mask2[:,:,sz2[2]-pxksz-1:sz2[2]-1]=0.
	
# ==========================================================================================================
# Compute HOG correlation between reprojected RMcube and a test frame
# ==========================================================================================================
   zmin2=0
   zmax2=0
   corrplane, corrcube, scube1, scube2 = HOGcorr_cube(galRMcube, np.array([hdu2[0].data]), zmin1, zmax1, zmin2, zmax2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1, mask2=mask2)

   plt.plot(velvec1, corrplane.ravel())	
   plt.xlabel('RM')
   plt.ylabel('Correlation')
   plt.show()
# ==========================================================================================================
# Compute HOG correlation between reprojected RM cube and Planck polarization
# ==========================================================================================================
   corrvec1, corrcube1, scube1 = HOGcorr_cubeandpol(galRMcube, ex, ey, zmin1, zmax1, ksz=ksz, mask1=mask1, mask2=mask2, rotatepol=True)

   #plt.plot(velvec1, corrvec0, 'r')
   plt.plot(velvec1, corrvec1, 'b')
   plt.xlabel(r'RM [rad m^-2]')
   plt.ylabel('PRS correlation')
   plt.show()

   imax=(corrvec1 == np.max(corrvec1)).nonzero()[0][0]

   ax1=plt.subplot(1,1,1, projection=WCS(hdu2[0].header))
   ax1.imshow(np.log10(galRMcube[imax,:,:]), origin='lower', cmap='rainbow') #, interpolation='none')
   ax1.coords.grid(color='white')
   ax1.coords['glon'].set_axislabel('Galactic Longitude')
   ax1.coords['glat'].set_axislabel('Galactic Latitude')
   ax1.coords['glat'].set_axislabel_position('r')
   ax1.coords['glat'].set_ticklabel_position('r')
   ax1.set_title('LOFAR RM')
   plt.show()

   import pdb; pdb.set_trace()

   strksz="%i" % ksz

   #plt.imshow(corrplane, origin='lower', extent=limsv/1e3)
   #plt.xlabel(r'$v_{CO}$ [km/s]')
   #plt.ylabel(r'$v_{HI}$ [km/s]')
   #plt.yticks(rotation='vertical')
   #plt.colorbar()
   #plt.savefig('HOGcorrelationPlanck353GRSL'+fstr+'_b'+blimstr+'_k'+strksz+'_v'+v1str+'to'+v2str+'.png', bbox_inches='tight')
   #plt.close()
   #import pdb; pdb.set_trace()

# --------------------------------------------------------------------------------------------------------
ksz=60 #arcsec 
astroHOGexampleGMIMS(23.75, 100., 135., ksz=ksz)


