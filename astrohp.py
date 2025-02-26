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

from statests import *

# -------------------------------------------------------------------------------------
def gaussian(x, mu, sig):

    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

# -------------------------------------------------------------------------------------
def astroHOGhp(map1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None):

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

   # ---------------------------------------------
   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(map1))
  
   if (lmax > lmax0):
      lmax=lmax0

   inmap1=map1.copy()-np.nanmean(map1)
   inmap2=map2.copy()-np.nanmean(map2)
 
   # Gradient of map 1 ---------------------------------------------
   #alm1=hp.sphtfunc.anafast(inmap1, iter=niter, alm=True, lmax=lmax, pol=False, use_weights=False, gal_cut=gal_cut, use_pixel_weights=False)
   #smap1, dmap1dtheta, dmap1dphi = hp.sphtfunc.alm2map_der1(alm1[1], hp.npix2nside(np.size(map1)), lmax=lmax, mmax=None) 
   alm1=hp.sphtfunc.map2alm(inmap1, iter=niter, use_pixel_weights=True)
   clm1=hp.sphtfunc.alm2cl(alm1)
   ell1=np.arange(np.size(clm1))+1

   g1=gaussian(np.arange(np.size(clm1)), 0., lmax)
   clip=g1/np.max(g1)
   #clip=np.ones(lmax+1)
   alm1_clipped=hp.almxfl(alm1, clip)
   clm1_clipped=hp.sphtfunc.alm2cl(alm1_clipped) 

   smap1, dmap1dtheta, dmap1dphi = hp.sphtfunc.alm2map_der1(alm1_clipped, hp.npix2nside(np.size(map1)))
   normdmap1=np.sqrt(dmap1dtheta**2+dmap1dphi**2)

   # Gradient of map 2 ------------------------------------------------------
   #alm2=hp.sphtfunc.anafast(inmap2, iter=niter, alm=True, lmax=lmax, pol=False, use_weights=False, gal_cut=gal_cut, use_pixel_weights=False)
   #smap2, dmap2dtheta, dmap2dphi = hp.sphtfunc.alm2map_der1(alm2[1], hp.npix2nside(np.size(map2)), lmax=lmax, mmax=None)
   alm2=hp.sphtfunc.map2alm(inmap2, iter=niter, use_pixel_weights=True)
   clm2=hp.sphtfunc.alm2cl(alm2)
   ell2=np.arange(np.size(clm2))+1

   clip=g1/np.max(g1)
   #clip=np.ones(lmax+1)
   alm2_clipped=hp.almxfl(alm2, clip)
   clm2_clipped=hp.sphtfunc.alm2cl(alm2_clipped)

   smap2, dmap2dtheta, dmap2dphi = hp.sphtfunc.alm2map_der1(alm2_clipped, hp.npix2nside(np.size(map2)))#, lmax=lmax)
   normdmap2=np.sqrt(dmap2dtheta**2+dmap2dphi**2)

   # Calculate relative orientation angles --------------------------------------------------------
   cosalpha=(dmap1dtheta*dmap2dtheta+dmap1dphi*dmap2dphi)/(normdmap1*normdmap2)
   sinalpha=(dmap1dtheta*dmap2dphi-dmap1dphi*dmap2dtheta)/(normdmap1*normdmap2)
   #alpha=np.arctan(sinalpha/cosalpha) 
   alpha=np.arctan2(sinalpha,cosalpha)

   alpha[(mask1 < 1.).nonzero()]=np.nan
   alpha[(mask2 < 1.).nonzero()]=np.nan

   index0=np.arange(0,np.size(inmap1),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)

   output=HOG_PRS(alpha[np.isfinite(alpha).nonzero()])
   Vall=output['Zx']   

   bookkeeping=np.zeros_like(inmap1)
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
 
   outmap1=smap1+np.nanmean(map1)  
   outmap2=smap2+np.nanmean(map2) 

   circstats={'Z': Zmap, 'V': Vmap, 'normdmap1': normdmap1, 'normdmap2': normdmap2, 'smap1': outmap1, 'smap2': outmap2, 'nmap': nangles, 'Vall': Vall}   

   return circstats 


