#!/usr/bin/env python
#
# This file is part of astroHOG
#
# CONTACT: juandiegosolerp[at]gmail.com
# Copyright (C) 2017-2023 Juan Diego Soler
#   
#------------------------------------------------------------------------------;

import sys
import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from congrid import *
from scipy import ndimage
from scipy import stats

import matplotlib.pyplot as plt

from nose.tools import assert_equal, assert_true

from statests import * 

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------------
def vprint(obj, verbose=True):
   if verbose:
      print(obj)
   return

# ---------------------------------------------------------------------------------------------------------------
def mse(x, y):
    return np.linalg.norm(x - y)

# ---------------------------------------------------------------------------------------------------------------
def imablockaverage(corrframe, nbx=7, nby=7, weight=1.):

   sz=np.shape(corrframe)
   limsx=np.linspace(0,sz[0]-1,nbx+1,dtype=int)
   limsy=np.linspace(0,sz[1]-1,nby+1,dtype=int)

   maxvblocks=np.zeros([nbx,nby])
   sigvblocks=np.zeros([nbx,nby])
   vblocks=np.zeros([nbx,nby])

   for i in range(0, np.size(limsx)-1):
      for k in range(0, np.size(limsy)-1):
         phi=corrframe[limsx[i]:limsx[i+1],limsy[k]:limsy[k+1]]
         tempphi=phi.ravel()
         wghts=0.*tempphi[np.isfinite(tempphi).nonzero()]+weight
         output=HOG_PRS(2.*tempphi[np.isfinite(tempphi).nonzero()], weights=wghts)
         vblocks[i,k]=output['Zx']

   imaxb, jmaxb = (vblocks==np.max(vblocks)).nonzero()

   return vblocks

# ------------------------------------------------------------------------------------------------------------------
def HOGcorr_ima(ima1, ima2, s_ima1=None, s_ima2=None, pxsz=1., ksz=1., res=1., nruns=0, mask1=None, mask2=None, gradthres1=None, gradthres2=None, weights=None, verbose=True):
   """ Calculates the spatial correlation between im1 and im2 using the HOG method and its confidence interval using Montecarlo sampling 

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

   sz1=np.shape(ima1)
   sz2=np.shape(ima2)

   if (s_ima1 is None):
      vprint('Warning: ima1 standard deviation not provided', verbose=verbose)
      mruns1=0
   else:
      if np.isscalar(s_ima1):
         vprint('Warning: common standard deviation provided for the whole map', verbose=verbose)
         s_ima1=np.copy(s_ima1)*np.ones_like(ima1)
      assert s_ima1.shape==ima1.shape, "Dimensions of s_ima1 and ima2 must match"
      mruns1=nruns

   if (s_ima2 is None):
      vprint('Warning: ima2 standard deviation not provided', verbose=verbose)
      mruns2=0
   else:
      if np.isscalar(s_ima2):
         vprint('Warning: common standard deviation provided for the whole map', verbose=verbose)
         s_ima2=np.copy(s_ima2)*np.ones_like(ima2)
      assert s_ima2.shape==ima2.shape, "Dimensions of s_ima2 and ima2 must match"
      mruns2=nruns

   # -----------------------------------------------
   ngoodvec=np.zeros(mruns1*mruns2)

   # Circular statistic outputs of orientation between image gradients
   rvec=np.zeros(mruns1*mruns2)
   zvec=np.zeros(mruns1*mruns2)
   vvec=np.zeros(mruns1*mruns2)
   vovervmaxvec=np.zeros(mruns1*mruns2)
   meanphivec=np.zeros(mruns1*mruns2)

    # Circular statistic outputs of directions between image gradients 
   rdvec=np.zeros(mruns1*mruns2)
   zdvec=np.zeros(mruns1*mruns2)
   vdvec=np.zeros(mruns1*mruns2)
   meanphidvec=np.zeros(mruns1*mruns2)   

   # Correlation statistics 
   pearvec=np.zeros(mruns1*mruns2)
   ccorvec=np.zeros(mruns1*mruns2)

   # Outputs ---------------------------------------------------------------------
   meanr=np.nan; s_r=np.nan;
   meanz=np.nan; s_z=np.nan;
   meanv=np.nan; s_v=np.nan; 
   meanphi=np.nan; s_meanphi=np.nan;
 
   meanrd=np.nan; s_rd=np.nan;
   meanzd=np.nan; s_zd=np.nan;
   meanvd=np.nan; s_vd=np.nan;
   meanphid=np.nan; s_meanphid=np.nan;
  
   meanpear=np.nan; s_meanpear=np.nan 
   meanccor=np.nan; s_meanccor=np.nan

   if (np.logical_or(mruns1 > 0, mruns2 > 0)):
      vprint("Running astroHOG Montecarlo ========================================", verbose=verbose)
      if (verbose):
         pbar = tqdm(total=mruns1*mruns2)

      for i in range(0,mruns1):
         rand1=np.random.normal(loc=ima1, scale=s_ima1)
         for k in range(0,mruns2):
            rand2=np.random.normal(loc=ima2, scale=s_ima2)

            circstats, corrframe, sima1, sima2 = HOGcorr_imaLITE(rand1, rand2, pxsz=pxsz, ksz=ksz, res=res, gradthres1=gradthres1, gradthres2=gradthres2, mask1=mask1, mask2=mask2, weights=weights)
            ind=np.ravel_multi_index((i, k),  dims=(mruns1,mruns2))

            ngoodvec[ind]=circstats['ngood']

            rvec[ind]=circstats['RVL']
            zvec[ind]=circstats['Z']
            vvec[ind]=circstats['V']
            vovervmaxvec[ind]=circstats['VoverVmax']
            meanphivec[ind]=circstats['meanphi']

            rdvec[ind]=circstats['RVLd']
            zdvec[ind]=circstats['Zd']
            vdvec[ind]=circstats['Vd']
            meanphidvec[ind]=circstats['meanphid']

            pearvec[ind]=circstats['pearsonr']
            ccorvec[ind]=circstats['crosscor']   

            if (verbose):
               pbar.update()
      
      if (verbose):
         pbar.close()  

      meanr=np.mean(rvec); s_r=np.std(rvec)
      meanz=np.mean(zvec); s_z=np.std(zvec)
      meanv=np.mean(vvec); s_v=np.std(vvec)
      meanvovervmax=np.mean(vovervmaxvec); s_vovervmax=np.std(vovervmaxvec)
      output=HOG_PRS(meanphivec)
      meanphi=output['meanphi']; s_meanphi=output['s_meanphi'];

      meanrd=np.mean(rdvec); s_rd=np.std(rdvec)
      meanzd=np.mean(zdvec); s_zd=np.std(zdvec)
      meanvd=np.mean(vdvec); s_vd=np.std(vdvec)
      output=HOG_PRS(meanphidvec)
      meanphid=output['meanphi']; s_meanphid=output['s_meanphi'];

      meanpear=np.mean(pearvec); s_pear=np.std(pearvec);     
      meanccor=np.mean(ccorvec); s_ccor=np.std(ccorvec);

      ngood=np.mean(ngoodvec)

   else:
      
      vprint('Montecarlo iterations disabled =============================', verbose=verbose)
      vprint('Warning: uncertainties on the correlation parameters will not be provided', verbose=verbose)
      circstats, corrframe, sima1, sima2 = HOGcorr_imaLITE(ima1, ima2, pxsz=pxsz, ksz=ksz, res=res, gradthres1=gradthres1, gradthres2=gradthres2, mask1=mask1, mask2=mask2, weights=weights)

      meanr=circstats['RVL']; s_r=np.nan
      meanz=circstats['Z'];   s_z=np.nan
      meanv=circstats['V'];   s_v=np.nan  
      meanvovervmax=circstats['VoverVmax'];   s_vovervmax=np.nan
      meanphi=circstats['meanphi']; s_meanphi=np.nan
 
      meanrd=circstats['RVLd']; s_rd=np.nan
      meanzd=circstats['Zd'];   s_zd=np.nan
      meanvd=circstats['Vd'];   s_vd=np.nan
      meanphid=circstats['meanphid']; s_meanphid=np.nan

      meanpear=circstats['pearsonr']; s_pear=np.nan
      meanccor=circstats['crosscor']; s_ccor=np.nan

      ngood=circstats['ngood']    

   circstats={'RVL': meanr, 'Z': meanz, 'V': meanv, 'VoverVmax': meanvovervmax, 'meanphi': meanphi,
              's_RVL': s_r, 's_Z': s_z, 's_V': s_v, 's_VoverVmax': s_vovervmax, 's_meanphi': s_meanphi,
              'RVLd': meanrd, 'Zd': meanzd, 'Vd': meanvd, 'meanphid': meanphid,
              's_RVLd': s_rd, 's_Zd': s_zd, 's_Vd': s_vd, 's_meanphid': s_meanphid,
              'pearsonr': meanpear, 's_pearsonr': s_pear, 'crosscor': meanccor, 's_crosscor': s_ccor, 
              'ngood': ngood}

   return circstats, corrframe, sima1, sima2


# ---------------------------------------------------------------------------------------------------------
def HOGcorr_imaLITE(ima1, ima2, pxsz=1., ksz=1., res=1., mode='nearest', mask1=None, mask2=None, gradthres1=None, gradthres2=None, weights=None, computejk=False, verbose=True):
   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
   ima1 : array corresponding to the first image to be compared 
   ima2 : array corresponding to the second image to be compared
   s_ima1 : 
   s_ima2 : 
   pxsz :
   ksz : Size of the derivative kernel in the pixel size units
   mode: Specify how the input array is extended when the kernel overlaps the border of the map. 
         Default: 'nearest'; The input is extended by replicating the last pixel.   
 
   Returns
   -------
    circstats:  Statistics describing the correlation between the input images.
                RVL      - Resulting vector lenght. 
                Z        - Rayleigh statistic.
                V        - Projected Ratleigh statistic.
                pearsonr - Pearson correlation coefficient.
                ngood    - Number of pixels used for the correlation
 
    corrframe : array containing the angles between the image gradients
    sima1     : ima1 smoothed with a 2D Gaussian with the size of the derivative kernel
    sima2     : ima2 smoothed with a 2D Gaussian with the size of the derivative kernel

   Examples
   --------

   """

   # Check if the images match
   assert ima2.shape == ima1.shape, "Dimensions of ima2 and ima1 must match"
   sz1=np.shape(ima1) 
   # Assign weights if none are specified
   if weights is None:
      weights=np.ones(sz1)
   # Assign weights if the weights are all the same
   if (np.size(weights)==1): 
      uniweights=weights
      weights=uniweights*np.ones(sz1)
   # Check if the provided weights match the image
   assert weights.shape == ima1.shape, "Dimensions of weights and ima1 must match" 

   # Check if the masks match the image shape
   if mask1 is None:
      "Mask 1 not defined by the user"
      mask1=np.ones_like(ima1)
   else:
      assert mask1.shape == ima1.shape, "Dimensions of mask1 and ima1 must match"
   if mask2 is None:
      "Mask 2 not defined by the user"
      mask2=np.ones_like(ima2)
   else:
      assert mask2.shape == ima2.shape, "Dimensions of mask2 and ima2 must match"

   pxksz=(ksz/(2*np.sqrt(2.*np.log(2.))))/pxsz #gaussian_filter takes sigma instead of FWHM as input

   # Calculate gradients
   sima1=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[0,0], mode=mode)
   sima2=ndimage.filters.gaussian_filter(ima2, [pxksz, pxksz], order=[0,0], mode=mode)
   dI1dx=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[0,1], mode=mode)
   dI1dy=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[1,0], mode=mode)
   dI2dx=ndimage.filters.gaussian_filter(ima2, [pxksz, pxksz], order=[0,1], mode=mode)
   dI2dy=ndimage.filters.gaussian_filter(ima2, [pxksz, pxksz], order=[1,0], mode=mode)

   # Calculation of the relative orientation angles
   phi=np.arctan2(dI1dx*dI2dy-dI1dy*dI2dx, dI1dx*dI2dx+dI1dy*dI2dy)
   #phi=np.arctan(np.tan(tempphi)) # Deprecated mapping to -90 to 90 range.

   # Excluding null gradients
   normGrad1=np.sqrt(dI1dx**2+dI1dy**2)
   normGrad2=np.sqrt(dI2dx**2+dI2dy**2)
   if np.logical_not(gradthres1 is None):
      bad=(normGrad1 <= gradthres1).nonzero()
      phi[bad]=np.nan
   if np.logical_not(gradthres2 is None):
      bad=(normGrad2 <= gradthres2).nonzero()
      phi[bad]=np.nan 
 
   # Excluding masked gradients
   if (np.size((mask1.ravel() > 0.).nonzero()) > 1):
      m1bad=(mask1 < 1.).nonzero()
      phi[m1bad]=np.nan
   else:
      vprint("No unmasked elements in ima1")
      phi[:]=np.nan
 
   if (np.size((mask2.ravel() > 0.).nonzero()) > 1):
      m2bad=(mask2 < 1.).nonzero()
      phi[m2bad]=np.nan
   else:
      vprint("No unmasked elements in ima2")
      phi[:]=np.nan

   if (np.size((mask1.ravel()*mask2.ravel() > 0.).nonzero()) < 1):
      vprint("No unmasked elements in the joint mask")
      phi[:]=np.nan

   good=np.isfinite(phi).nonzero()
   ngood=np.size(good)

   # Circular statistic outputs of orientation between image gradients
   rvl=np.nan # Resulting vector length (rvl)
   Z=np.nan; # Rayleigh statistic 
   V=np.nan; # Projected Rayleigh statistic
   VoverVmax=np.nan 
   meanphi=np.nan; # Mean orientation angle
   s_meanphi=np.nan;
 
   # Circular statistic outputs of directions between image gradients  
   rvld=np.nan # Resulting vector length (rvl)
   Zd=np.nan; # Rayleigh statistic 
   Vd=np.nan; # Projected Rayleigh statistic 
   meanphid=np.nan; # Mean orientation angle
   s_meanphid=np.nan;

   # Correlation statistics 
   pear=np.nan; # Pearson correlation coefficient 
   ccor=np.nan; # Crosscorrelation 

   if (ngood >= 2):

      # Calculate orientation statistics between image gradients 
      output=HOG_PRS(2.*phi[good], weights=weights[good])
      outputMax=HOG_PRS(2.*np.zeros_like(phi[good]), weights=weights[good])
      rvl=output['mrv']
      Z=output['Z']
      V=output['Zx']
      VoverVmax=output['Zx']/outputMax['Zx']

      s_V=output['s_Zx'] 
      meanphi=output['meanphi']
      ngood=output['ngood'] 
  
      # Calculate direction statistics between image gradients 
      output=HOG_PRS(phi[good], weights=weights[good])
      rvld=output['mrv']
      Zd=output['Z']
      Vd=output['Zx']
      s_Vd=output['s_Zx']
      meanphid=output['meanphi']     
 
      # Calculate Pearson correlation coefficient
      pear=PearsonCorrelationCoefficient(ima1[good], ima2[good])

      # Calculate cross correlation
      ccor=CrossCorrelation(ima1[good], ima2[good])

   else:

      vprint("WARNING: not enough pixels to compute astroHOG")

   circstats={'RVL': rvl, 'Z': Z, 'V': V, 'VoverVmax': VoverVmax, 'meanphi': meanphi, 
              'RVLd': rvld, 'Zd': Zd, 'Vd': Vd, 'meanphid': meanphid, 
	      'pearsonr': pear, 'crosscor': ccor, 'ngood': ngood}
   corrframe=phi

   return circstats, corrframe, sima1, sima2

# ---------------------------------------------------------------------------------------------------------
def HOGcorr_imaANDcube(ima1, cube2, pxsz=1., ksz=1., res=1., mode='nearest', mask1=None, mask2=None, gradthres1=None, gradthres2=None, weights=None, computejk=False, verbose=True, s_ima1=None, nruns=0):
   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
   ima1 : array corresponding to the first image to be compared 
   cube2 : array corresponding to the second image to be compared
   pxsz :
   ksz : Size of the derivative kernel in the pixel size units
   mode: Specify how the input array is extended when the kernel overlaps the border of the map. 
         Default: 'nearest'; The input is extended by replicating the last pixel.   
 
   Returns
   -------
    circstats:  Statistics describing the correlation between the input images.
                RVL      - Resulting vector lenght. 
                Z        - Rayleigh statistic.
                V        - Projected Ratleigh statistic.
                pearsonr - Pearson correlation coefficient.
                ngood    - Number of pixels used for the correlation
 
    corrframe : array containing the angles between the image gradients

    Examples
   --------

   """

   ima2=cube2[0,:,:]
 
   # Check if the images match ===================================================
   assert ima2.shape == ima1.shape, "Dimensions of ima2 and ima1 must match"
   sz1=np.shape(ima1)

   # Weights ====================================================================
   # Assign weights if none are specified
   if weights is None:
      weights=np.ones(sz1)
   # Assign weights if the weights are all the same
   if (np.size(weights)==1):
      uniweights=weights
      weights=uniweights*np.ones(sz1)
   # Check if the provided weights match the image
   assert weights.shape == ima1.shape, "Dimensions of weights and ima1 must match"

   # Check if the masks match the image shape
   if mask1 is None:
      "Mask 1 not defined by the user"
      mask1=np.ones_like(ima1)
   else:
      assert mask1.shape == ima1.shape, "Dimensions of mask1 and ima1 must match"

   imask2=mask2[0,:,:]
   if mask2 is None:
      "Mask 2 not defined by the user"
      mask2=np.ones_like(cube2)
   else:
      assert mask2.shape == cube2.shape, "Dimensions of mask2 and ima2 must match"

   pxksz=(ksz/(2*np.sqrt(2.*np.log(2.))))/pxsz #gaussian_filter takes sigma instead of FWHM as input

   # Calculate gradients of image 1
   sima1=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[0,0], mode=mode)
   dI1dx=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[0,1], mode=mode)
   dI1dy=ndimage.filters.gaussian_filter(ima1, [pxksz, pxksz], order=[1,0], mode=mode)

   sz2=np.shape(cube2)

   # Circular statistic outputs of orientation between image gradients
   vecRVL=np.nan*np.zeros(sz2[0]); # Resulting vector length (rvl)
   vecZ=np.nan*np.zeros(sz2[0]); # Rayleigh statistic 
   vecV=np.nan*np.zeros(sz2[0]); # Projected Rayleigh statistic
   vecs_V=np.nan*np.zeros(sz2[0]);
   vecVoverVmax=np.nan*np.zeros(sz2[0]);
   
   # Circular statistic outputs of directions between image gradients  
   vecRVLd=np.nan*np.zeros(sz2[0]) # Resulting vector length (rvl)
   vecZd=np.nan*np.zeros(sz2[0]); # Rayleigh statistic 
   vecVd=np.nan*np.zeros(sz2[0]); # Projected Rayleigh statistic 
   vecs_Vd=np.nan*np.zeros(sz2[0]);

   # Correlation statistics 
   vecpear=np.nan*np.zeros(sz2[0]); # Pearson correlation coefficient 
   vecccor=np.nan*np.zeros(sz2[0]); # Crosscorrelation 
 
   vecngood=np.nan*np.zeros(sz2[0]);

   scube2=np.nan*np.zeros_like(cube2)

   corrframe=np.nan*np.zeros([sz2[0],sz1[0],sz1[1]])   

   for i in range(0,sz2[0]):

      # Calculate gradients of images in cube2
      ima2=cube2[i,:,:]
      imask2=mask2[i,:,:]
      circstats12, corrframe12, sima1, sima2 = HOGcorr_ima(ima1, ima2, s_ima1=s_ima1, pxsz=pxsz, ksz=ksz, res=res, nruns=0, mask1=mask1, mask2=imask2, gradthres1=gradthres1, gradthres2=gradthres1, weights=weights, verbose=verbose)
 
      ngood=circstats12['ngood']

      if (ngood >= 2):

         # Calculate orientation statistics between image gradients 
         vecRVL[i]=circstats12['RVL']
         vecZ[i]=circstats12['Z']
         vecV[i]=circstats12['V']
         vecs_V[i]=circstats12['s_V']

         vecngood[i]=circstats12['ngood']
  
         # Calculate direction statistics between image gradients 
         vecRVLd[i]=circstats12['RVLd']
         vecZd[i]=circstats12['Zd']
         vecVd[i]=circstats12['Vd']
         vecs_Vd[i]=circstats12['s_Vd']
      
         # Calculate Pearson correlation coefficient
         vecpear[i]=circstats12['pearsonr']
         vecccor[i]=circstats12['crosscor']

      else:

         vprint("WARNING: not enough pixels to compute astroHOG")

   RVL=np.nanmean(vecRVL);             s_RVL=np.nanstd(vecRVL);
   Z=np.nanmean(vecZ);                 s_Z=np.nanstd(vecZ);
   V=np.nanmean(vecV);                 s_V=np.nanstd(vecV);
   
   RVLd=np.nanmean(vecRVLd);           s_RVLd=np.nanstd(vecRVLd);
   Zd=np.nanmean(vecZd);               s_Zd=np.nanstd(vecZd);
   Vd=np.nanmean(vecVd);               s_Vd=np.nanstd(vecVd);

   pearsonr=np.nanmean(vecpear)
   crosscor=np.nanmean(vecccor)

   circstats={'RVL': RVL, 'Z': Z, 'V': V, 
              's_RVL': s_RVL, 's_Z': s_Z, 's_V': s_V, 
              'RVLd': RVLd, 'Zd': Zd, 'Vd': Vd,
              's_RVLd': s_RVLd, 's_Zd': s_Zd, 's_Vd': s_Vd,
              'pearsonr': pearsonr, 'crosscor': crosscor,
              'vecngood': vecngood}

   return circstats, corrframe, sima1, sima2
   
# ----------------------------------------------------------------------------------------------------------------
def HOGcorr_frameandvec(frame1, vecx, vecy, gradthres=0., vecthres=0., pxsz=1., ksz=1., res=1., mask1=0, mask2=0, wd=1, allow_huge=False, regrid=False):
   # Calculates the spatial correlation between frame1 and the vector field described by vecx and vecy using the HOG methods
   #
   # INPUTS
   # frame1 - input map 
   # vecx   - x-component of the input vector field
   # vecy   - y-component of the input vector field
   #
   # OUTPUTS
   # hogcorr   -   
   # corrframe -

   sf=3. #Number of pixels per kernel FWHM      

   pxksz=(ksz/(2*np.sqrt(2.*np.log(2.))))/pxsz #gaussian_filter takes sigma instead of FWHM as input
   pxres =res/pxsz

   sz1=np.shape(frame1)

   if (ksz > 1):
      if (regrid):
         intframe1=congrid(frame1, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
         intvecx  =congrid(vecx,   [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
         intvecy  =congrid(vecy,   [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
         if np.array_equal(np.shape(frame1), np.shape(mask1)):
            intmask1=congrid(mask1, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
            intmask1[(intmask1 > 0.).nonzero()]=1.
            if np.array_equal(np.shape(frame2), np.shape(mask2)):
               intmask2=congrid(mask2, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
               intmask2[(intmask2 > 0.).nonzero()]=1.
      else:
         intframe1=frame1
         intvecx=vecx
         intvecy=vecy
         intmask1=mask1
         intmask2=mask2
      smoothframe1=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[0,0], mode='nearest')
      dI1dx=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[0,1], mode='nearest')
      dI1dy=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[1,0], mode='nearest')

   else:
      intframe1=frame1
      smoothframe1=frame1
      intvecx=vecx
      intvecy=vecy
      intmask1=mask1
      intmask2=mask2
      #grad1=np.gradient(intframe1)
      dI1dx=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[0,1], mode='nearest')
      dI1dy=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[1,0], mode='nearest')

   # ========================================================================================================================
   normGrad1=np.sqrt(dI1dx*dI1dx+dI1dy*dI1dy) #np.sqrt(grad1[1]**2+grad1[0]**2)
   normVec=np.sqrt(intvecx*intvecx + intvecy*intvecy)
   bad=np.logical_or(normGrad1 <= gradthres, normVec <= vecthres).nonzero()

   normGrad1[bad]=1.; normVec[bad]=1.;
   tempphi=np.arctan2(dI1dx*intvecy-dI1dy*intvecx, dI1dx*intvecx+dI1dy*intvecy)
   tempphi[bad]=np.nan
   phi=np.arctan(np.tan(tempphi))

   corrframe=np.cos(2.*phi)

   if np.array_equal(np.shape(intframe1), np.shape(intmask1)):
      corrframe[(intmask1 == 0.).nonzero()]=np.nan
      if np.array_equal(np.shape(intvecx), np.shape(intmask2)):
         corrframe[(intmask2 == 0.).nonzero()]=np.nan
         good=np.logical_and(np.logical_and(np.isfinite(phi), intmask1 > 0), intmask2 > 0).nonzero()
      else:
         good=np.logical_and(np.isfinite(phi), intmask1 > 0).nonzero()
   else:
         good=np.isfinite(phi).nonzero()
   #Zx, s_Zx, meanPhi = HOG_PRS(phi[good])
   output=HOG_PRS(2.*phi[good], w=weights[good])
   Zx=output['Zx']
   s_Zx=output['s_Zx']
   meanPhi=output['meanphi']

   return Zx, corrframe, smoothframe1



