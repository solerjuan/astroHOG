#!/usr/bin/env python
#
# This file is part of AstroHOG
#
# CONTACT: soler[AT]mpia.de
# Copyright (C) 2013-2019 Juan Diego Soler
#   
#------------------------------------------------------------------------------;

import sys
import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from scipy import ndimage

import pycircstat as circ
from nose.tools import assert_equal, assert_true

import matplotlib.pyplot as plt

import collections
import multiprocessing

from astrohog2d import *
from statests import *

from tqdm import tqdm

# ----------------------------------------------------------------------
def calculatexi(cosangles, nbins=20, thres=0.2):

   hist, bin_edges = np.histogram(np.abs(cosangles), bins=nbins, range=[0.0,1.0]) 
   bin_centres=0.5*(bin_edges[0:nbins]+bin_edges[1:nbins+1])  

   para=np.sum(hist[(bin_centres>=(1.0-thres)).nonzero()])
   perp=np.sum(hist[(bin_centres<=thres).nonzero()])
 
   xi=(para-perp)/(para+perp)
  
   return xi

# --------------------------------------------------------------------------------------------------------------------------------
def HOGcorr_cubeLITE(cube1, cube2, pxsz=1., ksz=1., mode='nearest', mask1=0., mask2=0., gradthres1=0., gradthres2=0., weights=None, weightbygradnorm=False):
   """ Calculates the spatial correlation between cube1 and cube2 using the HOG method 

   Parameters
   ----------   
   cube1 : array corresponding to the first  cube to be compared 
   cube2 : array corresponding to the second cube to be compared
   pxsz :  pixel size
   ksz :   size of the derivative kernel in the pixel size units
   
   Returns
   -------
    hogcorr :  
    corrframe :

   Examples
   --------

   """

   assert cube2.shape == cube1.shape, "Dimensions of cube2 and cube1 must match"
   sz1=np.shape(cube1)
   if weights is None:
      weights=np.ones(sz1)
   if (np.size(weights)==1):
      uniweights=weights
      weights=uniweights*np.ones(sz1)
   assert weights.shape == cube1.shape, "Dimensions of weights and ima1 must match"

   pxksz=(ksz/(2*np.sqrt(2.*np.log(2.))))/pxsz #gaussian_filter takes sigma instead of FWHM as input

   scube1=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,0,0], mode=mode)
   scube2=ndimage.filters.gaussian_filter(cube2, [pxksz, pxksz, pxksz], order=[0,0,0], mode=mode)

   dI1dx=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)
   dI1dy=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
   dI1dz=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)

   dI2dx=ndimage.filters.gaussian_filter(cube2, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)
   dI2dy=ndimage.filters.gaussian_filter(cube2, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
   dI2dz=ndimage.filters.gaussian_filter(cube2, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)

   normGrad1=np.sqrt(dI1dx**2+dI1dy**2+dI1dz**2)
   normGrad2=np.sqrt(dI2dx**2+dI2dy**2+dI2dz**2)

   # Calculation of the relative orientation angles
   #crossp=np.array([dI1dy*dI2dz-dI1dz*dI2dy,dI1dz*dI2dx-dI1dx*dI2dz,dI1dx*dI2dy-dI1dy*dI2dx])
   #crossp=np.sqrt(np.sum(crossp**2))
   dotp=dI1dx*dI2dx+dI1dy*dI2dy+dI1dz*dI2dz
   cosphi=dotp/(normGrad1*normGrad2)
   phi=np.arccos(cosphi)
 
   # Excluding null gradients
   bad=np.logical_or(normGrad1 <= gradthres1, normGrad2 <= gradthres2).nonzero()
   phi[bad]=np.nan

   # Excluding masked gradients
   if np.array_equal(np.shape(cube1), np.shape(mask1)):
      m1bad=(mask1 < 1.).nonzero()
      phi[m1bad]=np.nan
   if np.array_equal(np.shape(cube2), np.shape(mask2)):
      m2bad=(mask2 < 1.).nonzero()
      phi[m2bad]=np.nan

   good=np.isfinite(phi).nonzero()

   if (weightbygradnorm):
      weights=normGrad1*normGrad2
 
   Zx, s_Zx, meanPhi = HOG_PRS(phi[good])
   rvl=circ.descriptive.resultant_vector_length(2.*phi[good], w=weights[good])
   can=circ.descriptive.mean(2.*phi[good], w=weights[good])/2.
   pz, Z = circ.tests.rayleigh(2.*phi[good],  w=weights[good])
   pv, V = circ.tests.vtest(2.*phi[good], 0., w=weights[good])

   myV, s_myV, meanphi = HOG_PRS(2.*phi[good])

   am=HOG_AM(phi[good])

   pear, peap = stats.pearsonr(scube1[good], scube2[good])
   ngood=np.size(np.isfinite(phi.ravel()).nonzero())

   xi=calculatexi(cosphi[good])

   #circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am, pear, ngood, ssimv, msev]
   circstats = {'r': rvl, 'Z': Z, 'V': V, 'meanphi': meanphi, 'xi': xi}
   corrframe=phi

   return circstats, corrframe, scube1, scube2 

# -----------------------------------------------------------------------------------------------------------------------
def HOGcorr_cubeANDvecLITE(cube1, vec, pxsz=1., ksz=1., mode='nearest', mask1=0., mask2=0., gradthres1=0., gradthres2=0., weights=None):
   """ Calculates the correlation relative orientation between cube1 and the vector field 

   Parameters
   ----------   
   cube1 : array corresponding to the scale field cube
   vec :   array corresponding to the vector field [v_0,v_1,v_2], where v_i (i=0,1,2) corresponds to the i-th index of cube1
   pxsz :  pixel size
   ksz :   size of the derivative kernel in the pixel size units
   
   Returns
   -------
    hogcorr :  
    corrframe :

   Examples
   --------

   """

   assert vec1[0].shape == cube1.shape, "Dimensions of vec[0] and cube1 must match"
   assert vec1[1].shape == cube1.shape, "Dimensions of vec[1] and cube1 must match"
   assert vec1[2].shape == cube1.shape, "Dimensions of vec[2] and cube1 must match"

   scube1=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,0,0], mode=mode)
   dcube1d0=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)
   dcube1d1=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
   dcube1d2=ndimage.filters.gaussian_filter(cube1, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)

   crossp=np.array([dcube1d1*vec[2]-dcube1d2*vec[1],dcube1d2*vec[0]-dcube1d0*vec[2],dcube1d0*vec[1]-dcube1d1*vec[0]])
   normcrossp=np.sqrt(np.sum(crossp**2))
   dotp=dcube1d0*vec[0]+dcube1d1*vec[1]+dcube1d2*vec[2]
   tempphi=np.arctan2(normcrossp, dotp)
   phi=np.arctan(np.tan(tempphi))
 
   # Excluding null gradients
   normGrad1=np.sqrt(dcube1d0**2+dcube1d1**2+dcube1d2**2)
   normvec=np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
   bad=np.logical_or(normGrad1 <= gradthres1, normvec <= 0.0).nonzero()
   phi[bad]=np.nan

   # Excluding masked gradients
   if np.array_equal(np.shape(cube1), np.shape(mask1)):
      m1bad=(mask1 < 1.).nonzero()
      phi[m1bad]=np.nan
   if np.array_equal(np.shape(cube2), np.shape(mask2)):
      m2bad=(mask2 < 1.).nonzero()
      phi[m2bad]=np.nan
  
   good=np.isfinite(phi).nonzero()

   Zx, s_Zx, meanPhi = HOG_PRS(phi[good])
   rvl=circ.descriptive.resultant_vector_length(2.*phi[good], w=weights[good])
   can=circ.descriptive.mean(2.*phi[good], w=weights[good])/2.
   pz, Z = circ.tests.rayleigh(2.*phi[good],  w=weights[good])
   pv, V = circ.tests.vtest(2.*phi[good], 0., w=weights[good])

   myV, s_myV, meanphi = HOG_PRS(2.*phi[good])

   am=HOG_AM(phi[good])

   pear, peap = stats.pearsonr(scube1[good], scube2[good])

   ngood=np.size(good)

   ssimv=np.nan #ssim(sima1[good], sima2[good])
   msev =np.nan #mse(sima1[good], sima2[good])

   circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am, pear, ngood, ssimv, msev]
   corrframe=phi
 
   return circstats, corrframe, scube1 


