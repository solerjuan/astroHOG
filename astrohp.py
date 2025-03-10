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

from reproject import reproject_from_healpix, reproject_to_healpix

sigma2fwhm=2.*np.sqrt(2.*np.log(2.))

# -------------------------------------------------------------------------------------
def gaussian(x, mu, sig, limg=1e-3):

    gfunc=1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    gfunc[(gfunc < np.max(gfunc)*limg).nonzero()]=0.

    return gfunc

# -------------------------------------------------------------------------------------
def gradienthp(hpmap, niter=3, lmax=None, nsideout=None):

   if (lmax is None):
      lmax=2*np.npix2nside(np.size(hpmap))

   if (nsideout is None):
      nsideout=hp.npix2nside(np.size(hpmap))

   inhpmap=hpmap.copy()-np.nanmean(hpmap)
  
   #alm1=hp.sphtfunc.anafast(inmap1, iter=niter, alm=True, lmax=lmax, pol=False, use_weights=False, gal_cut=gal_cut, use_pixel_weights=False)
   #smap1, dmap1dtheta, dmap1dphi = hp.sphtfunc.alm2map_der1(alm1[1], hp.npix2nside(np.size(map1)), lmax=lmax, mmax=None)
   alm=hp.sphtfunc.map2alm(inhpmap, iter=niter, use_pixel_weights=True)
   clm=hp.sphtfunc.alm2cl(alm)
   ell=np.arange(np.size(clm))+1

   g1=gaussian(np.arange(np.size(clm)), 0., lmax)
   clip=g1/np.max(g1)
   #clip=np.ones(lmax+1)
   alm_clipped=hp.almxfl(alm, clip)
   clm_clipped=hp.sphtfunc.alm2cl(alm_clipped)

   smap, dmapdtheta, dmapdphi = hp.sphtfunc.alm2map_der1(alm_clipped, nsideout)
   gradmap=np.sqrt(dmapdtheta**2+dmapdphi**2)

   output={'dtheta': dmapdtheta, 'dphi': dmapdphi, 'smap': smap, 'gradmap': gradmap}

   return output

# -------------------------------------------------------------------------------------
def astroHOGhp(map1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True):

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

   # --------------------------------------------------------------
   resbase=hp.nside2resol(hp.npix2nside(np.size(map1)), arcmin=True)/60.0
   resHOG=hp.nside2resol(nsideout, arcmin=True)/60.0
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

   # Removing the mean ---------------------------------------------
   inmap1=map1.copy()-np.nanmean(map1)
   inmap2=map2.copy()-np.nanmean(map2)
 
   # Gradient of map 1 ---------------------------------------------
   output=gradienthp(map1, niter=niter, lmax=lmax)
   smap1=output['smap']
   dmap1dtheta=output['dtheta']
   dmap1dphi=output['dphi']
   normdmap1=output['gradmap']
 
   # Gradient of map 2 ------------------------------------------------------
   output=gradienthp(map2, niter=niter, lmax=lmax)
   smap2=output['smap']
   dmap2dtheta=output['dtheta']
   dmap2dphi=output['dphi']
   normdmap2=output['gradmap']

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

   if (computeVmap):
   
      for i in tqdm(index1):
 
         glon, glat = hp.pix2ang(nsideout, i, lonlat=True)
         target_header['CRVAL1']=glon
         target_header['CRVAL2']=glat  
 
         col1=fits.Column(name='I_STOKES', format='E', array=alpha)
         coldefs = fits.ColDefs([col1])
         hdu=fits.BinTableHDU.from_columns(coldefs)
         hdu.header['PIXTYPE']='HEALPIX'
         hdu.header['ORDERING']='RING' 
         hdu.header['COORDSYS']='G'
         subalpha, footprint = reproject_from_healpix(hdu, target_header)
         #hp.fitsfunc.write_map('dummy.fits', alpha, nest=False, coord='G', overwrite=True)
         #subalpha, footprint = reproject_from_healpix('dummy.fits', target_header)

         tempalpha=np.arctan(np.tan(np.abs(subalpha)))
         output=HOG_PRS(2.*tempalpha[np.isfinite(tempalpha).nonzero()])
         nangles[i]=np.size(np.isfinite(tempalpha).nonzero())
         Zmap[i]=output['Z']
         Vmap[i]=output['Zx'] 

   else:
      
      nangles[:]=0
      Zmap[:]=np.nan
      Vmap[:]=np.nan      

   outmap1=smap1+np.nanmean(map1)  
   outmap2=smap2+np.nanmean(map2) 

   circstats={'Z': Zmap, 'V': Vmap, 'normdmap1': normdmap1, 'normdmap2': normdmap2, 'smap1': outmap1, 'smap2': outmap2, 'nmap': nangles, 'Vall': Vall}   
   return circstats 

# -------------------------------------------------------------------------------------
def astroHOGhpSamples(samples1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True):

   nsamples, npix = np.shape(samples1)

   vecVall=np.zeros(nsamples)
   matVmap=np.zeros([nsamples,hp.nside2npix(nsideout)])

   for i in range(0,nsamples):

      output=astroHOGhp(samples1[i,:], map2, niter=niter, ksz=ksz, gal_cut=gal_cut, nsideout=nsideout, ordering1=ordering1, ordering2=ordering2, mask1=mask1, mask2=mask2, computeVmap=computeVmap)
      import pdb; pdb.set_trace()
      vecVall[i]=output['Vall']
      matVmap[i,:]=output['V']

   circstats={'V': matVmap, 'Vall': vecVall}

   return circstats

# -------------------------------------------------------------------------------------
def astroHOGhpPol(Imap, Qmap, Umap, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True):

   assert Imap.shape==Qmap.shape, "Dimensions of Imap and Qmap must match"
   assert Imap.shape==Umap.shape, "Dimensions of Imap and Umap must match"

   if (mask1 is None):
      mask1=np.ones_like(Imap)
   if (mask2 is None):
      mask2=np.ones_like(Qmap)
 
   # ---------------------------------------------
   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(Imap))

   if (lmax > lmax0):
      lmax=lmax0

   weights=((hp.nside2resol(hp.npix2nside(np.size(Imap)), arcmin=True)/60.)/ksz)**2

   # --------------------------------------------------------------
   resbase=hp.nside2resol(hp.npix2nside(np.size(Imap)), arcmin=True)/60.0
   resHOG=hp.nside2resol(nsideout, arcmin=True)/60.0
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

   # ------------------------------------------------------------------
   inImap=Imap.copy()-np.nanmean(Imap)
   inQmap=Qmap.copy()-np.nanmean(Qmap)
   inUmap=Umap.copy()-np.nanmean(Umap)

   # Gradient of map 1 ---------------------------------------------
   output=gradienthp(inImap, niter=niter, lmax=lmax)
   sImap=output['smap']
   dImapdtheta=output['dtheta']
   dImapdphi=output['dphi']
   gradImap=output['gradmap']

   # Gradient of map 2 ---------------------------------------------
   output=gradienthp(inQmap, niter=niter, lmax=lmax)
   sQmap=output['smap']
   dQmapdtheta=output['dtheta']
   dQmapdphi=output['dphi']
   gradQmap=output['gradmap']

   # Gradient of map 3 ---------------------------------------------
   output=gradienthp(inUmap, niter=niter, lmax=lmax)
   sUmap=output['smap']
   dUmapdtheta=output['dtheta']
   dUmapdphi=output['dphi']
   gradUmap=output['gradmap']

   # Calculating GradPoverP ----------------------------------------
   Pmap=np.sqrt(Qmap**2+Umap**2)
   gradPmap=np.sqrt(dQmapdtheta**2+dQmapdphi**2+dUmapdtheta**2+dUmapdphi**2)
   gradPoverPmap=gradPmap/Pmap

   dPoverPdtheta=np.sqrt(dQmapdtheta**2+dUmapdtheta**2)
   dPoverPdphi=np.sqrt(dQmapdphi**2+dUmapdphi**2)
   #output=gradienthp(gradPoverPmap, niter=niter, lmax=lmax)
   #dPoverPdphi=output['dtheta']
   #dPoverPdtheta=-output['dphi']
   #gradgradPoverPmap=np.sqrt(dPoverPdphi**2+dPoverPdtheta**2)

   cosalpha=(dImapdphi*dPoverPdphi+dImapdtheta*dPoverPdtheta)/(gradImap*gradPoverPmap)
   sinalpha=(dImapdphi*dPoverPdtheta-dImapdtheta*dPoverPdphi)/(gradImap*gradPoverPmap)
   #alpha=np.arctan(sinalpha/cosalpha)
   alpha=np.arctan2(sinalpha,cosalpha)

   alpha[(mask1 < 1.).nonzero()]=np.nan
   alpha[(mask2 < 1.).nonzero()]=np.nan

   output=HOG_PRS(alpha[np.isfinite(alpha).nonzero()])
   Vall=output['Zx'] 

   index0=np.arange(0,np.size(Imap),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)
 
   bookkeeping=np.zeros_like(Imap)
   nangles=np.zeros(hp.nside2npix(nsideout))
   Zmap=np.zeros(hp.nside2npix(nsideout))
   Vmap=np.zeros(hp.nside2npix(nsideout))

   if (computeVmap):

      for i in tqdm(index1):

         glon, glat = hp.pix2ang(nsideout, i, lonlat=True)
         target_header['CRVAL1']=glon
         target_header['CRVAL2']=glat 
         
         col1=fits.Column(name='I_STOKES', format='E', array=alpha)
         coldefs = fits.ColDefs([col1])
         hdu=fits.BinTableHDU.from_columns(coldefs)
         hdu.header['PIXTYPE']='HEALPIX'
         hdu.header['ORDERING']='RING'
         hdu.header['COORDSYS']='G'    
         subalpha, footprint = reproject_from_healpix(hdu, target_header)
 
         tempalpha=np.arctan(np.tan(np.abs(subalpha)))
         output=HOG_PRS(2.*tempalpha[np.isfinite(tempalpha).nonzero()])
         nangles[i]=np.size(np.isfinite(tempalpha).nonzero())
         Zmap[i]=output['Z']
         Vmap[i]=output['Zx'] 

   else:

      nangles[:]=0
      Zmap[:]=np.nan
      Vmap[:]=np.nan
 
   outmap1=sImap+np.nanmean(Imap)

   circstats={'Z': Zmap, 'V': Vmap, 'smap1': outmap1, 'psimap': alpha, 'gradImap': gradImap, 'gradPoverPmap': gradPoverPmap, 'Vall': Vall}

   return circstats

