# This file is part of AstroHOG
#
# Copyright (C) 2019-2025 Juan Diego Soler

import sys
import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from nose.tools import assert_equal, assert_true

import sklearn
import healpy as hp
from tqdm import tqdm

from astrohog2d import *
from statests import *

from reproject import reproject_from_healpix, reproject_to_healpix

sigma2fwhm=2.*np.sqrt(2.*np.log(2.0))

# -------------------------------------------------------------------------------------
def astroHOGhpwin(map1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None):

   """ Compares two healpix maps using the gradient orientation 

   Parameters
   ----------   
   ima1 : array corresponding to the first image to be compared 
   ima2 : array corresponding to the second image to be compared
   s_ima1 : 
   s_ima2 : 
   pxsz :
   ksz : Size of the derivative kernel in the pixel size units
   
   Returns
   -------
    hogcorr :  
    corrframe :

   Examples
   --------

   """

   assert map1.shape==map2.shape, "Dimensions of map1 and map2 must match"
  
   if (mask1 is None):
      mask1=np.ones_like(map1)
   if (mask2 is None):
      mask2=np.ones_like(map2)

   inmap1=map1.copy()-np.nanmean(map1)
   inmap2=map2.copy()-np.nanmean(map2)
 
   # ---------------------------------------------
   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(map1))
  
   if (lmax > lmax0):
      lmax=lmax0

   index0=np.arange(0,np.size(inmap1),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)

   resbase=(hp.nside2resol(hp.npix2nside(np.size(inmap1)), arcmin=True)/60.0).tolist()
   resHOG=(hp.nside2resol(nsideout, arcmin=True)/60.0).tolist() 

   pxksz=(ksz/resbase)/sigma2fwhm

   hdu=fits.PrimaryHDU()
   hdu.header['NAXIS']=2
   hdu.header['NAXIS1']=int(resHOG/resbase)
   hdu.header['NAXIS2']=int(resHOG/resbase)
   hdu.header['CTYPE1']='GLON-TAN'
   hdu.header['CRPIX1']=hdu.header['NAXIS1']/2
   hdu.header['CRVAL1']=0.
   hdu.header['CDELT1']=-resbase
   hdu.header['CUNIT1']='deg     '
   hdu.header['CTYPE2']='GLAT-TAN'
   hdu.header['CRPIX2']=hdu.header['NAXIS2']/2
   hdu.header['CRVAL2']=0.
   hdu.header['CDELT2']=resbase
   hdu.header['CUNIT2']='deg     '
   hdu.header['COORDSYS']='Galactic'
   target_header=hdu.header.copy()

   bookkeeping=np.zeros_like(inmap1)
   nangles=np.zeros(hp.nside2npix(nsideout))
   Zmap=np.zeros(hp.nside2npix(nsideout))
   Vmap=np.zeros(hp.nside2npix(nsideout))

   for i in index1:

      glon, glat = hp.pix2ang(nsideout, i, lonlat=True)

      target_header['CRVAL1']=glon
      target_header['CRVAL2']=glat   
 
      hp.fitsfunc.write_map('dummy.fits', inmap1, nest=False, coord='G', overwrite=True)
      submap1, footprint = reproject_from_healpix('dummy.fits', target_header)

      hp.fitsfunc.write_map('dummy.fits', inmap2, nest=False, coord='G', overwrite=True)
      submap2, footprint = reproject_from_healpix('dummy.fits', target_header)

      circstats, corrframe, sima1, sima2 = HOGcorr_ima(submap1, submap2, ksz=pxksz, verbose=False)
      Zmap[i]=circstats['Z']
      Vmap[i]=circstats['V']
      nangles[i]=circstats['ngood']

   circstats={'Z': Zmap, 'V': Vmap, 'nangles': nangles}

   return circstats 

# -------------------------------------------------------------------------------------
def astroHOGhpwinPol(Imap, Qmap, Umap, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None):
 
   # ---------------------------------------------
   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(Imap))

   if (lmax > lmax0):
      lmax=lmax0

   inImap=Imap.copy()-np.nanmean(Imap)
   inQmap=Qmap.copy()-np.nanmean(Qmap)
   inUmap=Umap.copy()-np.nanmean(Umap)

   output=gradienthp(inImap, niter=niter, lmax=lmax)
   sImap=output['smap']
   dImapdtheta=output['dtheta']
   dImapdphi=output['dphi']
   gradImap=output['gradmap']

   output=gradienthp(inQmap, niter=niter, lmax=lmax)
   sQmap=output['smap']
   dQmapdtheta=output['dtheta']
   dQmapdphi=output['dphi']
   gradQmap=output['gradmap']

   output=gradienthp(inUmap, niter=niter, lmax=lmax)
   sUmap=output['smap']
   dUmapdtheta=output['dtheta']
   dUmapdphi=output['dphi']
   gradUmap=output['gradmap']

   Pmap=np.sqrt(Qmap**2+Umap**2)
   gradPmap=np.sqrt(dQmapdtheta**2+dQmapdphi**2+dUmapdtheta**2+dUmapdphi**2)
   gradPoverPmap=gradPmap/Pmap

   output=gradienthp(gradPoverPmap, niter=niter, lmax=lmax)
   dPoverPdphi=output['dtheta']
   dPoverPdtheta=-output['dphi']
   gradgradPoverPmap=np.sqrt(dPoverPdphi**2+dPoverPdtheta**2)

   cosalpha=(dImapdphi*dPoverPdphi+dImapdtheta*dPoverPdtheta)/(gradImap*gradPoverPmap)
   sinalpha=(dImapdphi*dPoverPdtheta-dImapdtheta*dPoverPdphi)/(gradImap*gradPoverPmap)
   #alpha=np.arctan(sinalpha/cosalpha)
   alpha=np.arctan2(sinalpha,cosalpha)

   output=HOG_PRS(alpha[np.isfinite(alpha).nonzero()])
   Vall=output['Zx']  

   index0=np.arange(0,np.size(Imap),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)
 
   bookkeeping=np.zeros_like(Imap)
   nangles=np.zeros(hp.nside2npix(nsideout))
   Zmap=np.zeros(hp.nside2npix(nsideout))
   Vmap=np.zeros(hp.nside2npix(nsideout))

   for i in index1:

      theta, phi = hp.pix2ang(nsideout, i, lonlat=False)
      ipix=hp.query_disc(hp.npix2nside(np.size(bookkeeping)), hp.ang2vec(theta, phi), hp.nside2resol(nsideout))
      bookkeeping[ipix]+=1.
      tempalpha=alpha[ipix] 
      output=HOG_PRS(tempalpha[np.isfinite(tempalpha).nonzero()])
      nangles[i]=np.size(np.isfinite(tempalpha).nonzero())
      Zmap[i]=output['Z']
      Vmap[i]=output['Zx']

   circstats={'Z': Zmap, 'V': Vmap, 'gradImap': gradImap, 'gradPoverPmap': gradPoverPmap, 'Vall': Vall}

   return circstats

