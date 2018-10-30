#
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
def HOGthreekpcarm(vmin, vmax, ksz=1, suffix='', glonmin=12., glonmax=-12., MakePlots=False):

   # Define the cube limits
   v1=vmin; v2=vmax;
   v1str="%4.1f" % vmin
   v2str="%4.1f" % vmax
   limsv=np.array([v1, v2, v1, v2])
   strksz="%i" % ksz

   # ------------------------------------------------------------------------------------
   # Read cubes 
   dir1='/Users/soler/Documents/PYTHON/ThreeKpcArm/' 
   hdu1=fits.open(dir1+'hogDHT02_Center_bw_mom.fits')
   cube1=hdu1[0].data
   refhdr1=hdu1[0].header.copy()
   hdu1.close()

   cube1[np.isnan(cube1)]=0.

   sz1=np.shape(cube1)
   zmin1=int(refhdr1['CRPIX3']+(v1-refhdr1['CRVAL3'])/refhdr1['CDELT3'])
   zmax1=int(refhdr1['CRPIX3']+(v2-refhdr1['CRVAL3'])/refhdr1['CDELT3'])
   velvec1=refhdr1['CRVAL3']+refhdr1['CDELT3']*(np.arange(refhdr1['NAXIS3'])-refhdr1['CRPIX3'])   #np.arange(v1,v2,refhdr1['CDELT3'])

   # Produce 2D reference header
   REFhdr=refhdr1.copy() 
   del REFhdr['NAXIS3']
   del REFhdr['CTYPE3']
   del REFhdr['CDELT3']
   del REFhdr['CRPIX3']
   del REFhdr['CRVAL3']
   REFhdr['NAXIS']=2

   # ------------------------------------------------------------------------------------- 
   # Read cubes
   dir2='/Users/soler/Documents/PYTHON/ThreeKpcArm/'
   hdu2=fits.open(dir2+'redline_b1_for_juan.fits')
   rawcube2=hdu2[0].data
   refhdr2=hdu2[0].header.copy()
   cube2=np.transpose(rawcube2,axes=(2,0,1))
   hdu2.close() 
   cube2[np.isnan(cube2)]=0.

   # Produce 2D reference header and reorder the header    
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

   sz2=np.shape(cube2)
   zmin2=0  #int(refhdr2['CRPIX3']+(v1-refhdr2['CRVAL3'])/refhdr2['CDELT3'])
   zmax2=refhdr2['NAXIS1']-1 #int(refhdr2['CRPIX3']+(v2-refhdr2['CRVAL3'])/refhdr2['CDELT3'])
   velvec2=refhdr2['CRVAL1']+refhdr2['CDELT1']*(np.arange(refhdr2['NAXIS1'])-refhdr2['CRPIX1'])

   if (MakePlots):

      # Show images
      plt.figure(figsize = (18,2))
      plt.subplot(projection=WCS(TARhdr))
      plt.imshow(cube1[zmin1:zmax1,:,:].sum(axis=0), origin='lower', cmap='hot')
      plt.show()

      plt.figure(figsize = (18,2))
      plt.subplot(projection=WCS(TARhdr))
      plt.imshow(cube2[zmin2:zmax2,:,:].sum(axis=0), origin='lower', cmap='terrain')
      plt.show()
      #import pdb; pdb.set_trace()

   # ----------------------------------------------------------------
   limsv=np.array([velvec2[zmin2], velvec2[zmax2], velvec1[zmin1], velvec1[zmax1]])
   mywcs=WCS(TARhdr)

   px, py = mywcs.wcs_world2pix(glonmin, -1., 1)
   minl=int(px)
   minb=int(py)
   px, py = mywcs.wcs_world2pix(glonmax, 1., 1)
   maxl=int(px)
   maxb=int(py)

   temp1=cube1[:,minb:maxb,minl:maxl]
   temp2=cube2[:,minb:maxb,minl:maxl]

   mask1=1.0+0.0*temp1
   mask2=1.0+0.0*temp2

   # Define mask1
   rmsI1=np.sort(cube1.std(axis=(1,2)))
   thres1=rmsI1[(rmsI1 > 0.).nonzero()][0] 
   mask1[(temp1 < thres1).nonzero()]=0.
   mask1[:,0:2*ksz,:]=0.; mask1[:,sz1[1]-2*ksz:sz1[1],:]=0.
   mask1[:,:,0:2*ksz]=0.; mask1[:,:,sz1[2]-2*ksz:sz1[2]]=0.
 
   # Define mask2 
   rmsI2=np.sort(cube2.std(axis=(1,2)))
   #thres2=rmsI2[(rmsI2 > 0.).nonzero()][0]  
   thres2=0. # No masking the dust maps
   mask2[(temp2 < thres2).nonzero()]=0.
   mask2[:,0:2*ksz,:]=0.; mask2[:,sz2[1]-2*ksz:sz2[1],:]=0.
   mask2[:,:,0:2*ksz]=0.; mask2[:,:,sz2[2]-2*ksz:sz2[2]]=0.

   corrplane, corrcube, scube1, scube2 =HOGcorr_cube(temp1, temp2, zmin1, zmax1-1, zmin2, zmax2-1, ksz=ksz, mask1=mask1, mask2=mask2)
   vplane =corrplane[2]

   szout=np.shape(vplane)
   xv, yv = np.meshgrid(np.arange(szout[0]), np.arange(szout[1]), sparse=False, indexing='ij')

   wts=vplane.copy()
   wts[np.isnan(wts).nonzero()]=0.
   colPRS=int(np.sum(wts*xv)/np.sum(wts))
   rowPRS=int(np.sum(wts*yv)/np.sum(wts))  

   maxPRS=np.max(vplane[np.isfinite(vplane).nonzero()])
   maxind=(vplane == maxPRS).nonzero()
   colPRS=maxind[0][0] 
   rowPRS=maxind[1][0]

   print('Max PRS at:')
   print('v_CO=',velvec1[zmin1+colPRS],' km/s') 
   print('d   =',velvec2[zmin2+rowPRS],' kpc')
   limsg=np.array([np.max([glonmin,glonmax]),np.min([glonmin,glonmax]),-1, 1.])

   if (MakePlots):

      fig = plt.figure(figsize=[2.0, 4.2], dpi=150)
      plt.rc('font', size=4)
      ax1=plt.subplot(311)
      im1=ax1.imshow(temp1[zmin1+colPRS,:,:], origin='lower', extent=limsg)
      cbar=fig.colorbar(im1, fraction=0.085, pad=0.04, ax=ax1)
      cbar.ax.set_title(r'$I$ [K]')
      ax1.set_title('v_CO='+str(velvec1[zmin1+colPRS])+' km/s')
      ax2=plt.subplot(312, sharex=ax1)
      im2=ax2.imshow(temp2[zmin2+rowPRS,:,:], origin='lower', extent=limsg)
      cbar=fig.colorbar(im2, fraction=0.085, pad=0.04, ax=ax2)
      cbar.ax.set_title(r'$A_{K}$ [mag]')
      ax2.set_title('d='+str(velvec2[zmin2+rowPRS])+' kpc')
      ax3=plt.subplot(313, sharex=ax1)
      im=ax3.imshow((180./np.pi)*np.abs(corrcube[colPRS, rowPRS,:,:]), origin='lower', cmap='spring', extent=limsg)
      cbar=fig.colorbar(im, fraction=0.085, pad=0.04, ax=ax3)
      cbar.ax.set_title(r'$|\phi|$')
      plt.tight_layout()
      plt.show()

      fig = plt.figure(figsize=[2.5, 2.5], dpi=150)
      plt.rc('font', size=4) 
      im=plt.imshow(vplane, origin='lower', extent=limsv, clim=[0.,np.max(vplane[np.isfinite(vplane)])], aspect='auto')
      plt.title('GLOT=['+str(glonmin)+','+str(glonmax)+']')
      plt.ylabel(r'$v_{CO}$ [km/s]')
      plt.xlabel(r'$d$ [kpc]')
      cbar=fig.colorbar(im, fraction=0.085, pad=0.04)
      cbar.ax.set_title(r'PRS')
      plt.tight_layout()
      plt.show()

   #import pdb; pdb.set_trace()

   return velvec1[zmin1+colPRS], velvec2[zmin2+rowPRS]

# --------------------------------------------------------------------------------------------------------
ksz=1 #arcsec

vco0, d0 = HOGthreekpcarm(-120., 0., ksz=ksz, glonmin=12.,  glonmax=9., MakePlots=True)

vco1, d1 = HOGthreekpcarm(-120., 0., ksz=ksz, glonmin=12.,  glonmax=9.)
vco2, d2 = HOGthreekpcarm(-120., 0., ksz=ksz, glonmin=9.,   glonmax=6.)
vco3, d3 = HOGthreekpcarm(-120., 0., ksz=ksz, glonmin=-1.,  glonmax=-4.)
vco4, d4 = HOGthreekpcarm(-120., 20., ksz=ksz, glonmin=-4.,  glonmax=-7.)
vco5, d5 = HOGthreekpcarm(-120., 20., ksz=ksz, glonmin=-7.,  glonmax=-10.)
vco6, d6 = HOGthreekpcarm(-120., 20., ksz=ksz, glonmin=-10., glonmax=-12.)

vels=[vco1,vco2,vco3,vco4,vco5,vco6]
dist=[d1,d2,d3,d4,d5,d6]

fig = plt.figure(figsize=[2.5, 2.5], dpi=150)
plt.rc('font', size=4)
plt.plot(dist,vels,'ro')
plt.xlabel(r'$d$ [kpc]')
plt.ylabel(r'$v_{CO}$ [km/s]')
plt.text(dist[0], vels[0], '[12,9]')
plt.text(dist[1], vels[1], '[9,6]')
plt.text(dist[2], vels[2], '[-1,-4]')
plt.text(dist[3], vels[3], '[-4,-7]')
plt.text(dist[4], vels[4], '[-7,-10]')
plt.text(dist[5], vels[5], '[-10,-12]')
plt.tight_layout()
plt.show()


