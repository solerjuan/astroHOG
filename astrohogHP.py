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
fwhm2sigma=1/sigma2fwhm


def circular_mean(angles, axis=None):
    """
    Compute the circular mean of an array of angles (in radians)
    along a specified axis.

    Parameters
    ----------
    angles : np.ndarray
        Array of angles in radians.
    axis : int or None
        Axis along which to compute the mean. If None, compute over
        the entire array.

    Returns
    -------
    np.ndarray
        Circular mean of the angles in radians.
    """
    angles = np.asarray(angles)

    # Convert angles to complex numbers on the unit circle
    complex_repr = np.exp(1j * angles)

    # Average in the complex plane
    mean_complex = np.mean(complex_repr, axis=axis)

    # Convert back to angle
    return np.angle(mean_complex)

# -------------------------------------------------------------------------------------
def gaussian(x, mu, sig, limg=1e-3):

    gfunc=1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    gfunc[(gfunc < np.max(gfunc)*limg).nonzero()]=0.

    return gfunc

# -------------------------------------------------------------------------------------
def gradienthp(hpmap, niter=3, ksz=None, w0=0., nsideout=None, ordering='ring'):

   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
    hpmap   : healpix map 
    niter   : 
    ksz     : Size of the derivative kernel in degrees
    nsideout: nside of the output map
 
   Returns
   -------
    circstats:  Statistics describing the correlation between the input images.

   """

   if (ksz is None):
      ksz=2*np.rad2deg(hp.nside2resol(hp.npix2nside(np.size(hpmap))))

   ksz1=np.sqrt(ksz**2-w0**2)

   lmax=int(np.ceil(180/ksz))

   if (nsideout is None):
      nsideout=hp.npix2nside(np.size(hpmap))
 
   if (ordering=='nested'):
      inhpmap=hp.reorder(hpmap, n2r=True)-np.nanmean(hpmap)
   else:
      inhpmap=hpmap.copy()-np.nanmean(hpmap)
  
   #alm1=hp.sphtfunc.anafast(inmap1, iter=niter, alm=True, lmax=lmax, pol=False, use_weights=False, gal_cut=gal_cut, use_pixel_weights=False)
   #smap1, dmap1dtheta, dmap1dphi = hp.sphtfunc.alm2map_der1(alm1[1], hp.npix2nside(np.size(map1)), lmax=lmax, mmax=None)
   
   #alm=hp.sphtfunc.map2alm(inhpmap, iter=niter) 
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
def astroHOGhpStride(map1, map2, niter=3, ksz=3.0, gal_cut=0, vsize=np.rad2deg(hp.nside2resol(8)), ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True, weights=None, w1=0., w2=0., s_map1=None, s_map2=None):

   nsidein1=hp.npix2nside(np.size(map1))
   nsidein2=hp.npix2nside(np.size(map2))
   nsidein=np.min([nsidein1,nsidein2])

   if (mask1 is None):
      mask1=np.ones_like(map1)
   if (mask2 is None):
      mask2=np.ones_like(map2)

   if (nsidein1 > nsidein2):
      mask1=hp.pixelfunc.ud_grade(mask1.copy(), nsidein)
   if (nsidein1 < nsidein2):
      mask2=hp.pixelfunc.ud_grade(mask2.copy(), nsidein)

   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(map1))

   if (lmax > lmax0):
      lmax=lmax0

   if (weights is None):
      weights=((hp.nside2resol(hp.npix2nside(np.size(map1)), arcmin=True)/60.)/ksz)**2

   resbase=hp.nside2resol(hp.npix2nside(np.size(map1)), arcmin=True)/60.0
   resHOG=hp.nside2resol(nsidein, arcmin=True)/60.0
   pxksz=(ksz/resbase)/sigma2fwhm

   # Gradient of map 1 ---------------------------------------------
   output=gradienthp(map1, niter=niter, ksz=ksz, w0=w1, nsideout=nsidein, ordering=ordering1)
   smap1=output['smap']
   dmap1dtheta=output['dtheta']
   dmap1dphi=output['dphi']
   gradmap1=output['gradmap']

   # Gradient of map 2 ------------------------------------------------------
   output=gradienthp(map2, niter=niter, ksz=ksz, w0=w2, nsideout=nsidein, ordering=ordering2)
   smap2=output['smap']
   dmap2dtheta=output['dtheta']
   dmap2dphi=output['dphi']
   gradmap2=output['gradmap']

   # Calculate relative orientation angles --------------------------------------------------------
   cosalpha=(dmap1dtheta*dmap2dtheta+dmap1dphi*dmap2dphi)/(gradmap1*gradmap2)
   sinalpha=(dmap1dtheta*dmap2dphi-dmap1dphi*dmap2dtheta)/(gradmap1*gradmap2)
   alphao=np.arctan(sinalpha/cosalpha)
   alphad=np.arctan2(sinalpha,cosalpha)

   alphao[(mask1 < 1.).nonzero()]=np.nan;
   alphao[(mask2 < 1.).nonzero()]=np.nan
   alphad[(mask1 < 1.).nonzero()]=np.nan;
   alphad[(mask2 < 1.).nonzero()]=np.nan

   output=HOG_PRS(2.*alphao[np.isfinite(alphao).nonzero()], weights=weights)
   Voall=output['Zx']
   output=HOG_PRS(alphad[np.isfinite(alphad).nonzero()], weights=weights)
   Vdall=output['Zx']

   # Blocks ----------------------------------------------------------------------------------
   nsidein=hp.npix2nside(np.size(alphad))
   bookkeeping=np.zeros_like(alphad)

   # -----------------------------------------------------------------
   nangles=np.zeros(hp.nside2npix(nsidein))
   Zomap=np.zeros(hp.nside2npix(nsidein))
   Zdmap=np.zeros(hp.nside2npix(nsidein))

   Vomap=np.zeros(hp.nside2npix(nsidein))
   Vdmap=np.zeros(hp.nside2npix(nsidein))

   VoMAXmap=np.zeros(hp.nside2npix(nsidein))
   VdMAXmap=np.zeros(hp.nside2npix(nsidein))

   meanmap1=np.zeros(hp.nside2npix(nsidein))
   meanmap2=np.zeros(hp.nside2npix(nsidein))
   stdmap1=np.zeros(hp.nside2npix(nsidein))
   stdmap2=np.zeros(hp.nside2npix(nsidein))

   npix=np.size(map1)
   nside1=hp.npix2nside(npix)
   indices=np.arange(0,npix,1)
   vectors=np.asarray(hp.pix2vec(nsidein, indices)).T

   npix2=np.size(map2)
   nside2=hp.npix2nside(npix2)
   radius=np.deg2rad(0.5*vsize) #hp.nside2resol(nsideout)

   # ----------------------------
   valid = np.isfinite(alphad)

   cos_d = np.zeros_like(alphad)
   sin_d = np.zeros_like(alphad)
   cos_o = np.zeros_like(alphad)
   sin_o = np.zeros_like(alphad)

   cos_d[valid] = np.cos(alphad[valid])
   sin_d[valid] = np.sin(alphad[valid])

   # Axial orientation statistic
   cos_o[valid]=np.cos(2.0*alphad[valid])
   sin_o[valid]=np.sin(2.0*alphad[valid])

   weight = float(np.ravel(weights)[0])  # Current default is scalar

   for i in tqdm(indices):

      selpix2=hp.query_disc(nside2, vectors[i,:], radius) 
      meanmap2[i]=np.nanmean(map2[selpix2])
      stdmap2[i]=np.nanstd(map2[selpix2])

      selpix=hp.query_disc(nside1, vectors[i,:], radius)
      meanmap1[i]=np.nanmean(map1[selpix])
      stdmap1[i]=np.nanstd(map1[selpix])

      # -----------------------------------------------------
      goodpix = selpix[valid[selpix]]
      ngood=goodpix.size
      denominator=np.sqrt(ngood*weight/2.0)

      cd=weight*np.sum(cos_d[goodpix])
      sd=weight*np.sum(sin_d[goodpix])
      co=weight*np.sum(cos_o[goodpix])
      so=weight*np.sum(sin_o[goodpix])

      Vdmap[i] = cd / denominator
      Zdmap[i] = np.hypot(cd, sd) / denominator

      Vomap[i] = co / denominator
      Zomap[i] = np.hypot(co, so) / denominator

      VdMAXmap[i]=np.sqrt(2.0*ngood*weight)
      VoMAXmap[i]=VdMAXmap[i]
      nangles[i]=ngood

   circstats={'Vdall': Vdall, 'Voall': Voall, 'nmap': nangles,
              'Vd': Vdmap, 'VdMAX': VdMAXmap,
              'Vo': Vomap, 'VoMAX': VoMAXmap,
              'alphad': alphad, 'alphao': alphao,
              'meanmap1': meanmap1, 'meanmap2': meanmap2,
              'stdmap1': stdmap1, 'stdmap2': stdmap2,
              'gradmap1': gradmap1, 'gradmap2': gradmap2,
              'smap1': smap1, 'smap2': smap2}

   return circstats

# -------------------------------------------------------------------------------------
def astroHOGhplite(map1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True, weights=None, w1=0., w2=0., s_map1=None, s_map2=None):

   nsidein1=hp.npix2nside(np.size(map1))
   nsidein2=hp.npix2nside(np.size(map2))
   nsidein=np.min([nsidein1,nsidein2])

   if (mask1 is None):
      mask1=np.ones_like(map1)
   if (mask2 is None):
      mask2=np.ones_like(map2)

   if (nsidein1 > nsidein2):
      mask1=hp.pixelfunc.ud_grade(mask1.copy(), nsidein)
   if (nsidein1 < nsidein2):
      mask2=hp.pixelfunc.ud_grade(mask2.copy(), nsidein)

   lmax=int(180./ksz)
   lmax0=hp.npix2nside(np.size(map1))

   if (lmax > lmax0):
      lmax=lmax0

   if (weights is None):
      weights=((hp.nside2resol(hp.npix2nside(np.size(map1)), arcmin=True)/60.)/ksz)**2

   resbase=hp.nside2resol(hp.npix2nside(np.size(map1)), arcmin=True)/60.0
   resHOG=hp.nside2resol(nsideout, arcmin=True)/60.0
   pxksz=(ksz/resbase)/sigma2fwhm

   # Gradient of map 1 ---------------------------------------------
   output=gradienthp(map1, niter=niter, ksz=ksz, w0=w1, nsideout=nsidein, ordering=ordering1)
   smap1=output['smap']
   dmap1dtheta=output['dtheta']
   dmap1dphi=output['dphi']
   gradmap1=output['gradmap']

   # Gradient of map 2 ------------------------------------------------------
   output=gradienthp(map2, niter=niter, ksz=ksz, w0=w2, nsideout=nsidein, ordering=ordering2)
   smap2=output['smap']
   dmap2dtheta=output['dtheta']
   dmap2dphi=output['dphi']
   gradmap2=output['gradmap']

   # Calculate relative orientation angles --------------------------------------------------------
   cosalpha=(dmap1dtheta*dmap2dtheta+dmap1dphi*dmap2dphi)/(gradmap1*gradmap2)
   sinalpha=(dmap1dtheta*dmap2dphi-dmap1dphi*dmap2dtheta)/(gradmap1*gradmap2)
   alphao=np.arctan(sinalpha/cosalpha)
   alphad=np.arctan2(sinalpha,cosalpha)

   alphao[(mask1 < 1.).nonzero()]=np.nan;
   alphao[(mask2 < 1.).nonzero()]=np.nan
   alphad[(mask1 < 1.).nonzero()]=np.nan;
   alphad[(mask2 < 1.).nonzero()]=np.nan

   output=HOG_PRS(2.*alphao[np.isfinite(alphao).nonzero()], weights=weights)
   Voall=output['Zx']
   output=HOG_PRS(alphad[np.isfinite(alphad).nonzero()], weights=weights)
   Vdall=output['Zx']

   # Blocks ----------------------------------------------------------------------------------
   nsidein=hp.npix2nside(np.size(alphad))

   index0=np.arange(0,np.size(alphad),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)

   bookkeeping=np.zeros_like(alphad)

   # -----------------------------------------------------------------
   nangles=np.zeros(hp.nside2npix(nsideout))
   Zomap=np.zeros(hp.nside2npix(nsideout))
   Zdmap=np.zeros(hp.nside2npix(nsideout))

   Vomap=np.zeros(hp.nside2npix(nsideout))
   Vdmap=np.zeros(hp.nside2npix(nsideout))

   VoMAXmap=np.zeros(hp.nside2npix(nsideout))
   VdMAXmap=np.zeros(hp.nside2npix(nsideout))

   meanmap1=np.zeros(hp.nside2npix(nsideout))
   meanmap2=np.zeros(hp.nside2npix(nsideout))
   stdmap1=np.zeros(hp.nside2npix(nsideout))
   stdmap2=np.zeros(hp.nside2npix(nsideout))

   for i in index1:
      
      dummy=np.zeros(hp.nside2npix(nsideout))

      selpix=hp.query_disc(nsidein, hp.pixelfunc.pix2vec(nsideout,i), 0.5*hp.nside2resol(nsideout))

      meanmap1[i]=np.nanmean(map1[selpix])
      meanmap2[i]=np.nanmean(map2[selpix])
      stdmap1[i]=np.nanstd(map1[selpix])
      stdmap2[i]=np.nanstd(map2[selpix])

      tempalphad=alphad[selpix]
      output=HOG_PRS(0.*tempalphad[np.isfinite(tempalphad).nonzero()], weights=weights)
      VdMAXmap[i]=output['Zx']
      output=HOG_PRS(tempalphad[np.isfinite(tempalphad).nonzero()], weights=weights)
      nangles[i]=np.size(np.isfinite(tempalphad).nonzero())
      Zdmap[i]=output['Z']
      Vdmap[i]=output['Zx']

      tempalphao=alphao[selpix]
      output=HOG_PRS(2.*tempalphao[np.isfinite(tempalphao).nonzero()])
      Zomap[i]=output['Z']
      Vomap[i]=output['Zx']
      output=HOG_PRS(np.zeros(np.size(tempalphad)), weights=weights)
      VoMAXmap[i]=output['Zx']

   circstats={'Vdall': Vdall, 'Voall': Voall, 'nmap': nangles,
              'Vd': Vdmap, 'VdMAX': VdMAXmap,
              'Vo': Vomap, 'VoMAX': VoMAXmap,
              'alphad': alphad, 'alphao': alphao,
              'meanmap1': meanmap1, 'meanmap2': meanmap2,
              'stdmap1': stdmap1, 'stdmap2': stdmap2,
              'gradmap1': gradmap1, 'gradmap2': gradmap2,
              'smap1': smap1, 'smap2': smap2}

   return circstats

# -------------------------------------------------------------------------------------
def astroHOGhp(map1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True, weights=None, w1=0., w2=0., s_map1=None, s_map2=None, nruns=1):

   nsidein1=hp.npix2nside(np.size(map1))
   nsidein2=hp.npix2nside(np.size(map2))
   nsidein=np.min([nsidein1,nsidein2])

   Voall=np.nan; s_Voall=np.nan
   Vdall=np.nan; s_Vdall=np.nan
   Vomap=np.nan; s_Vomap=np.nan
   Vdmap=np.nan; s_Vdmap=np.nan 
   VoMAXmap=np.nan; s_VoMAXmap=np.nan
   VdMAXmap=np.nan; s_VdMAXmap=np.nan
   alphao=np.nan; alphad=np.nan
   meanmap1=np.nan; s_meanmap1=np.nan
   meanmap2=np.nan; s_meanmap2=np.nan
   nangles=np.nan

   # ==========================================================================
   if (nruns > 1):
 
      Voallvec=np.zeros(nruns)
      Vdallvec=np.zeros(nruns)
      Vomapvec=np.zeros([nruns,hp.nside2npix(nsideout)])
      Vdmapvec=np.zeros([nruns,hp.nside2npix(nsideout)])
      VoMAXmapvec=np.zeros([nruns,hp.nside2npix(nsideout)])
      VdMAXmapvec=np.zeros([nruns,hp.nside2npix(nsideout)])
      meanmap1vec=np.zeros([nruns,hp.nside2npix(nsideout)])
      meanmap2vec=np.zeros([nruns,hp.nside2npix(nsideout)])
      stdmap1vec=np.zeros([nruns,hp.nside2npix(nsideout)])
      stdmap2vec=np.zeros([nruns,hp.nside2npix(nsideout)])
      nanglesvec=np.zeros([nruns,hp.nside2npix(nsideout)])

      gradmap1vec=np.zeros([nruns,hp.nside2npix(nsidein)])
      gradmap2vec=np.zeros([nruns,hp.nside2npix(nsidein)])
      alphaovec=np.zeros([nruns,hp.nside2npix(nsidein)])
      alphadvec=np.zeros([nruns,hp.nside2npix(nsidein)])
 
      print(" astroHOGhp: Monte Carlo realizations")
      for i in tqdm(range(0,nruns)):

         rmap2=np.random.normal(loc=map2, scale=s_map2)

         output=astroHOGhplite(map1, rmap2, niter=niter, ksz=ksz, nsideout=nsideout, ordering1=ordering1, ordering2=ordering2, mask1=mask1, mask2=mask2, computeVmap=computeVmap, weights=weights, w1=w1, w2=w2, s_map1=s_map1, s_map2=s_map2)

         Voallvec[i]=output['Voall']
         Vdallvec[i]=output['Vdall']

         Vomapvec[i,:]=output['Vo']
         Vdmapvec[i,:]=output['Vd']
         VoMAXmapvec[i,:]=output['VoMAX']
         VdMAXmapvec[i,:]=output['VdMAX']
         meanmap1vec[i,:]=output['meanmap1']
         meanmap2vec[i,:]=output['meanmap2']
         stdmap1vec[i,:]=output['stdmap1']
         stdmap2vec[i,:]=output['stdmap2']

         gradmap1vec[i,:]=output['gradmap1']
         gradmap2vec[i,:]=output['gradmap2']
         nanglesvec[i,:]=output['nmap']
         alphaovec[i,:]=output['alphao']
         alphadvec[i,:]=output['alphad'] 
  
      Voall=np.mean(Voallvec)
      s_Voall=np.std(Voallvec)
      Vdall=np.mean(Vdallvec)
      s_Vdall=np.std(Vdallvec)
  
      Vomap=np.mean(Vomapvec, axis=0)
      s_Vomap=np.std(Vomapvec, axis=0)      
      Vdmap=np.mean(Vdmapvec, axis=0)
      s_Vdmap=np.std(Vdmapvec, axis=0)
      VoMAXmap=np.mean(VoMAXmapvec, axis=0)
      s_VoMAXmap=np.std(VoMAXmapvec, axis=0)
      VdMAXmap=np.mean(VdMAXmapvec, axis=0)
      s_VdMAXmap=np.std(VdMAXmapvec, axis=0)

      meanmap1=np.mean(meanmap1vec, axis=0)
      s_meanmap1=np.std(meanmap1vec, axis=0)
      stdmap1=np.mean(stdmap1vec, axis=0)
      meanmap2=np.mean(meanmap2vec, axis=0)
      s_meanmap2=np.std(meanmap2vec, axis=0)
      stdmap2=np.mean(stdmap2vec, axis=0)
      nangles=np.mean(nanglesvec, axis=0)
  
      gradmap1=np.mean(gradmap1vec, axis=0)
      gradmap2=np.mean(gradmap2vec, axis=0)
 
      alphao=circular_mean(alphaovec, axis=0)
      alphad=circular_mean(alphadvec, axis=0)

   else:
   
      output=astroHOGhplite(map1, map2, niter=niter, ksz=ksz, nsideout=nsideout, ordering1=ordering1, ordering2=ordering2, mask1=mask1, mask2=mask2, computeVmap=computeVmap, weights=weights, w1=w1, w2=w2, s_map1=s_map1, s_map2=s_map2)   

      Voall=output['Voall']
      Vdall=output['Vdall'] 
      Vomap=output['Vo']
      Vdmap=output['Vd']
      alphao=output['alphao']
      alphad=output['alphad']
      VoMAXmap=output['VoMAX']
      VdMAXmap=output['VdMAX']
      meanmap1=output['meanmap1']; meanmap2=output['meanmap2']
      stdmap1=output['stdmap1']; stdmap2=output['stdmap2']
      gradmap1=output['gradmap1']; gradmap2=output['gradmap2']
      nangles=output['nmap']
      alphao=output['alphao']
      alphad=output['alphad']
 
   circstats={'Vdall': Vdall, 'Voall': Voall,
              's_Vdall': s_Vdall, 's_Voall': s_Voall, 
              'nmap': nangles,
              'Vd': Vdmap, 's_Vd': s_Vdmap,'VdMAX': VdMAXmap,
              'Vo': Vomap, 's_Vo': s_Vomap, 'VoMAX': VoMAXmap,
              'alphad': alphad, 'alphao': alphao, 
              'meanmap1': meanmap1, 'meanmap2': meanmap2,
              'stdmap1': stdmap1, 'stdmap2': stdmap2,
              'gradmap1': gradmap1, 'gradmap2': gradmap2}   
 
   return circstats 

# -------------------------------------------------------------------------------------
def astroHOGhpSamples(samples1, map2, niter=3, ksz=3.0, gal_cut=0, nsideout=8, ordering1='ring', ordering2='ring', mask1=None, mask2=None, computeVmap=True, weights=None, w1=0., w2=0., s_map1=None, s_map2=None, nruns=1):

   nsidein1=hp.npix2nside(np.shape(samples1)[1])
   nsidein2=hp.npix2nside(np.size(map2))
   nsidein=np.min([nsidein1,nsidein2])

   nsamples, npix = np.shape(samples1)

   vecVall=np.zeros(nsamples)
   matVmap=np.zeros([nsamples,hp.nside2npix(nsideout)])

   Voallvec=np.zeros(nsamples)
   Vdallvec=np.zeros(nsamples)
   Vomapvec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   Vdmapvec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   VoMAXmapvec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   VdMAXmapvec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   meanmap1vec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   meanmap2vec=np.zeros([nsamples,hp.nside2npix(nsidein)])
   stdmap1vec =np.zeros([nsamples,hp.nside2npix(nsidein)])
   stdmap2vec =np.zeros([nsamples,hp.nside2npix(nsidein)])
   nanglesvec =np.zeros([nsamples,hp.nside2npix(nsidein)])
   alphaovec  =np.zeros([nsamples,hp.nside2npix(nsidein)])
   alphadvec  =np.zeros([nsamples,hp.nside2npix(nsidein)])

   for i in range(0,nsamples):

      output=astroHOGhp(samples1[i,:], map2, niter=niter, ksz=ksz, gal_cut=gal_cut, nsideout=nsideout, ordering1=ordering1, ordering2=ordering2, mask1=mask1, mask2=mask2, computeVmap=computeVmap, weights=weights, w1=w1, w2=w2, s_map1=s_map1, s_map2=s_map2, nruns=nruns)
      Voallvec[i]=output['Voall']
      Vdallvec[i]=output['Vdall']

      Vomapvec[i,:]=output['Vo']
      Vdmapvec[i,:]=output['Vd']
      VoMAXmapvec[i,:]=output['VoMAX']
      VdMAXmapvec[i,:]=output['VdMAX']
      meanmap1vec[i,:]=output['meanmap1']
      meanmap2vec[i,:]=output['meanmap2']
      stdmap1vec[i,:]=output['stdmap1']
      stdmap2vec[i,:]=output['stdmap2']

      nanglesvec[i,:]=output['nmap']
      alphaovec[i,:]=output['alphao']
      alphadvec[i,:]=output['alphad']
 
   Voall=np.mean(Voallvec)
   s_Voall=np.std(Voallvec)
   Vdall=np.mean(Vdallvec)
   s_Vdall=np.std(Vdallvec)

   Vomap=np.mean(Vomapvec, axis=0)
   s_Vomap=np.std(Vomapvec, axis=0)
   Vdmap=np.mean(Vdmapvec, axis=0)
   s_Vdmap=np.std(Vdmapvec, axis=0)
   VoMAXmap=np.mean(VoMAXmapvec, axis=0)
   s_VoMAXmap=np.std(VoMAXmapvec, axis=0)
   VdMAXmap=np.mean(VdMAXmapvec, axis=0)
   s_VdMAXmap=np.std(VdMAXmapvec, axis=0)

   meanmap1=np.mean(meanmap1vec, axis=0)
   s_meanmap1=np.std(meanmap1vec, axis=0)
   stdmap1=np.mean(stdmap1vec, axis=0)
   meanmap2=np.mean(meanmap2vec, axis=0)
   s_meanmap2=np.std(meanmap2vec, axis=0)
   stdmap2=np.mean(stdmap2vec, axis=0)
   nangles=np.mean(nanglesvec, axis=0)

   alphao=circular_mean(alphaovec, axis=0)
   alphad=circular_mean(alphadvec, axis=0)

   circstats={'Vdall': Vdall, 's_Vdall': Vdall, 
              'Voall': Voall, 's_Voall': Voall,
              'Vd': Vdmap, 's_Vd': s_Vdmap,
              'Vo': Vomap, 's_Vo': s_Vomap,
              'nmap': nangles,
              'VdMAX': VdMAXmap, 'VoMAX': VoMAXmap,
              'alphad': alphad, 'alphao': alphao,
              'meanmap1': meanmap1, 'meanmap2': meanmap2,
              'stdmap1': stdmap1, 'stdmap2': stdmap2}

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

   Pmap=np.sqrt(Qmap**2+Umap**2) 
   # Calculating GradPsi ------------------------------------------
   output=gradienthp(Qmap/Pmap, niter=niter, lmax=lmax)
   dQoverPdtheta=output['dtheta']
   dQoverPdphi=output['dphi']

   output=gradienthp(Umap/Pmap, niter=niter, lmax=lmax)
   dUoverPdtheta=output['dtheta']
   dUoverPdphi=output['dphi']

   dpsidtheta=np.sqrt(dQoverPdtheta**2+dUoverPdtheta**2)
   dpsidphi=np.sqrt(dQoverPdphi**2+dUoverPdphi**2)
   gradpsi=np.sqrt(dQoverPdtheta**2+dUoverPdtheta**2+dQoverPdphi**2+dUoverPdphi**2)

   cosalpha=(dImapdphi*dpsidphi+dImapdtheta*dpsidtheta)/(gradImap*gradpsi)
   sinalpha=(dImapdphi*dpsidtheta-dImapdtheta*dpsidphi)/(gradImap*gradpsi)
   alphao=np.arctan(sinalpha/cosalpha)
   alphad=np.arctan2(sinalpha,cosalpha)

   alphao[(mask1 < 1.).nonzero()]=np.nan; alphao[(mask2 < 1.).nonzero()]=np.nan
   alphad[(mask1 < 1.).nonzero()]=np.nan; alphad[(mask2 < 1.).nonzero()]=np.nan

   # Calculating GradPoverP ----------------------------------------
   Pmap=np.sqrt(Qmap**2+Umap**2)
   gradPmap=np.sqrt(dQmapdtheta**2+dQmapdphi**2+dUmapdtheta**2+dUmapdphi**2)
   gradPoverPmap=gradPmap/Pmap

   dPoverPdtheta=np.sqrt(dQmapdtheta**2+dUmapdtheta**2)
   dPoverPdphi=np.sqrt(dQmapdphi**2+dUmapdphi**2)

   #cosalpha=(dImapdphi*dPoverPdphi+dImapdtheta*dPoverPdtheta)/(gradImap*gradPoverPmap)
   #sinalpha=(dImapdphi*dPoverPdtheta-dImapdtheta*dPoverPdphi)/(gradImap*gradPoverPmap)
   #alpha=np.arctan2(sinalpha,cosalpha)

   # ---------------------------------------------------------------
   alphao[(mask1 < 1.).nonzero()]=np.nan; alphao[(mask2 < 1.).nonzero()]=np.nan
   alphad[(mask1 < 1.).nonzero()]=np.nan; alphad[(mask2 < 1.).nonzero()]=np.nan

   output=HOG_PRS(2.*alphao[np.isfinite(alphao).nonzero()])
   Voall=output['Zx']
   output=HOG_PRS(alphad[np.isfinite(alphad).nonzero()])
   Vdall=output['Zx']

   index0=np.arange(0,np.size(Imap),1)
   index1=np.arange(0,hp.nside2npix(nsideout),1)

   bookkeeping=np.zeros_like(Imap)
   nangles=np.zeros(hp.nside2npix(nsideout))

   Zomap=np.zeros(hp.nside2npix(nsideout))
   Zdmap=np.zeros(hp.nside2npix(nsideout))

   Vomap=np.zeros(hp.nside2npix(nsideout))
   Vdmap=np.zeros(hp.nside2npix(nsideout))

   if (computeVmap):

      for i in tqdm(index1):

         glon, glat = hp.pix2ang(nsideout, i, lonlat=True)
         target_header['CRVAL1']=glon
         target_header['CRVAL2']=glat
         glonvec=(np.arange(target_header['NAXIS1'])-target_header['CRPIX1'])*target_header['CDELT1']+target_header['CRVAL1']
         glatvec=(np.arange(target_header['NAXIS2'])-target_header['CRPIX2'])*target_header['CDELT2']+target_header['CRVAL2']
         
         poly=hp.pixelfunc.ang2vec(np.array([glonvec[0],glonvec[0],glonvec[-1],glonvec[-1]]), np.array([glatvec[0],glatvec[-1],glatvec[-1],glatvec[0]]), lonlat=True)
         selpix=hp.query_polygon(hp.npix2nside(np.size(alphad)), poly)

         #col1=fits.Column(name='I_STOKES', format='E', array=alpha)
         #coldefs = fits.ColDefs([col1])
         #hdu=fits.BinTableHDU.from_columns(coldefs)
         #hdu.header['PIXTYPE']='HEALPIX'
         #hdu.header['ORDERING']='RING' 
         #hdu.header['COORDSYS']='G'
         #subalpha, footprint = reproject_from_healpix(hdu, target_header)

         tempalphad=alphad[selpix]
         output=HOG_PRS(tempalphad[np.isfinite(tempalphad).nonzero()])
         nangles[i]=np.size(np.isfinite(tempalphad).nonzero())
         Zdmap[i]=output['Z']
         Vdmap[i]=output['Zx']

         tempalphao=alphao[selpix]
         output=HOG_PRS(2.*tempalphao[np.isfinite(tempalphao).nonzero()])
         Zomap[i]=output['Z']
         Vomap[i]=output['Zx']

   else:

      nangles[:]=0
      Zdmap[:]=np.nan
      Vdmap[:]=np.nan
      Zomap[:]=np.nan
      Vomap[:]=np.nan         
 
   outmap1=sImap+np.nanmean(Imap)

   circstats={'Vdall': Vdall, 'Voall': Voall, 'nmap': nangles,
              'Zd': Zdmap, 'Vd': Vdmap,
              'Zo': Zomap, 'Vo': Vomap,
              'alphad': alphad, 'alphao': alphao, 
              'smap1': outmap1, 'gradImap': gradImap, 'gradPoverPmap': gradPoverPmap, 'gradPsi': gradpsi}

   return circstats

