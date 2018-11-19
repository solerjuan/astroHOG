#!/usr/bin/env python
#
# This file is part of AstroHOG
#
# CONTACT: soler[AT]mpia.de
# Copyright (C) 2013-2018 Juan Diego Soler
#   
#------------------------------------------------------------------------------;

import sys
import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from congrid import *
from scipy import ndimage

import pycircstat as circ
from nose.tools import assert_equal, assert_true

from statests import * 

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------------------------------
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
         pz, Zx = circ.tests.vtest(2.*tempphi[np.isfinite(tempphi).nonzero()],0.,w=wghts)
         vblocks[i,k] = Zx

   imaxb, jmaxb = (vblocks==np.max(vblocks)).nonzero()

   return vblocks

# --------------------------------------------------------------------------------------------------------------------------------
def HOGcorr_ima(ima1, ima2, s_ima1=0., s_ima2=0., ksz=1., nruns=10, mask1=0., mask2=0.):

   sz1=np.shape(ima1)
   sz2=np.shape(ima2)

   mruns1=nruns
   if (s_ima1 == 0.):
      mruns1=1

   mruns2=nruns
   if (s_ima2 == 0.):
      mruns2=1

   rvec=np.zeros(mruns1*mruns2)
   zvec=np.zeros(mruns1*mruns2)
   vvec=np.zeros(mruns1*mruns2)
   amvec=np.zeros(mruns1*mruns2) 

   for i in range(0,mruns1):
      rand1=np.random.normal(loc=ima1, scale=s_ima1+0.*ima1)
      for k in range(0,mruns2):
         rand2=np.random.normal(loc=ima2, scale=s_ima1+0.*ima1)         
         circstats, corrframe, sima1, sima2=HOGcorr_imaLITE(rand1, rand2, ksz=ksz, mask1=mask1, mask2=mask2)
         rvec[np.ravel_multi_index((i, k), dims=(mruns1,mruns2))] =circstats[0]
         zvec[np.ravel_multi_index((i, k), dims=(mruns1,mruns2))] =circstats[1]
         vvec[np.ravel_multi_index((i, k), dims=(mruns1,mruns2))] =circstats[2]
         amvec[np.ravel_multi_index((i, k), dims=(mruns1,mruns2))]=circstats[8]

   circstats, corrframe, sima1, sima2=HOGcorr_imaLITE(ima1, ima2, ksz=ksz)
   outr=circstats[0]
   outv=circstats[2]

   #circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am]
   meanr=np.mean(rvec)
   meanz=np.mean(zvec)
   meanv=np.mean(vvec)
   s_r  =np.std(rvec)
   s_z  =np.std(zvec)
   s_v  =np.std(vvec)
   am   =np.mean(amvec)
   s_am =np.std(amvec)
   circstats=[meanr, meanz, meanv, s_r, s_z, s_v, outr, outv, am, s_am]

   #import matplotlib.pyplot as plt
   #plt.hist(vvec)
   #plt.show()
   #import pdb; pdb.set_trace()
   return circstats, corrframe, sima1, sima2


# --------------------------------------------------------------------------------------------------------------------------------
def HOGcorr_imaLITE(ima1, ima2, ksz=1., mode='nearest', mask1=0., mask2=0.):
   # Calculates the spatial correlation between im1 and im2 using the HOG method
   #
   # INPUTS
   # ima1 -
   # ima2 -
   # ksz -	Size of the derivative kernel in pixel units
   #
   # OUTPUTS
   # hogcorr -   
   # corrframe -

   sima1=ndimage.filters.gaussian_filter(ima1, [ksz, ksz], order=[0,0], mode=mode)
   sima2=ndimage.filters.gaussian_filter(ima2, [ksz, ksz], order=[0,0], mode=mode)
   dI1dx=ndimage.filters.gaussian_filter(ima1, [ksz, ksz], order=[0,1], mode=mode)
   dI1dy=ndimage.filters.gaussian_filter(ima1, [ksz, ksz], order=[1,0], mode=mode)
   dI2dx=ndimage.filters.gaussian_filter(ima2, [ksz, ksz], order=[0,1], mode=mode)
   dI2dy=ndimage.filters.gaussian_filter(ima2, [ksz, ksz], order=[1,0], mode=mode)

   # Calculation of the relative orientation angles
   tempphi=np.arctan2(dI1dx*dI2dy-dI1dy*dI2dx, dI1dx*dI2dx+dI1dy*dI2dy)
   phi=np.arctan(np.tan(tempphi))

   # Excluding null gradients
   normGrad1=np.sqrt(dI1dx**2+dI1dy**2)
   normGrad2=np.sqrt(dI2dx**2+dI2dy**2)
   bad=np.logical_or(normGrad1 == 0., normGrad2 == 0.).nonzero()
   
   # Excluding masked gradients
   if np.array_equal(np.shape(ima1), np.shape(mask1)):
      m1bad=(mask1 < 1.).nonzero()
      phi[m1bad]=np.nan
   if np.array_equal(np.shape(ima2), np.shape(mask2)):
      m2bad=(mask2 < 1.).nonzero()
      phi[m2bad]=np.nan

   good=np.isfinite(phi).nonzero()

   Zx, s_Zx, meanPhi = HOG_PRS(phi[good])

   weight=(1./ksz)**2
   wghts=0.*phi[good]+weight

   rvl=circ.descriptive.resultant_vector_length(2.*phi[good], w=wghts)
   can=circ.descriptive.mean(2.*phi[good], w=wghts)/2.
   pz, Z = circ.tests.rayleigh(2.*phi[good],  w=wghts)
   pv, V = circ.tests.vtest(2.*phi[good], 0., w=wghts)

   myV, s_myV, meanphi = HOG_PRS(2.*phi[good])

   am = HOG_AM(phi[good])

   circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am]
   corrframe=phi
   
   return circstats, corrframe, sima1, sima2


# -------------------------------------------------------------------------------------------------------------------------------
def HOGcorr_frame(frame1, frame2, gradthres1=0., gradthres2=0., pxsz=1., ksz=1., res=1., mask1=0, mask2=0, wd=1, allow_huge=False, regrid=False):
   # Calculates the spatial correlation between frame1 and frame2 using the HOG method
   #
   # INPUTS
   # frame1 -
   # frame2 -
   # gradthres1 -
   # gradthres2 -
   # pxsz -
   # ksz -
   # res -
   # mask1 -
   # mask2 -
   # wd -
   # regrid -
   #
   # OUTPUTS
   # hogcorr -   
   # corrframe -

   sf=3. #Number of pixels per kernel FWHM      

   pxksz =ksz/pxsz
   pxres =res/pxsz

   sz1=np.shape(frame1)
   sz2=np.shape(frame1)

   if (ksz > 1):
      weight=(pxsz/ksz)**2

      if (regrid):
         intframe1=congrid(frame1, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
         intframe2=congrid(frame2, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
         if np.array_equal(np.shape(frame1), np.shape(mask1)):
            intmask1=congrid(mask1, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
            intmask1[(intmask1 > 0.).nonzero()]=1.
            if np.array_equal(np.shape(frame2), np.shape(mask2)):
               intmask2=congrid(mask2, [np.int(np.round(sf*sz1[0]/pxres)), np.int(np.round(sf*sz1[1]/pxres))])
               intmask2[(intmask2 > 0.).nonzero()]=1.
      else:
         intframe1=frame1
         intframe2=frame2
         intmask1=mask1
         intmask2=mask2
      smoothframe1=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[0,0], mode='nearest')
      smoothframe2=ndimage.filters.gaussian_filter(frame2, [pxksz, pxksz], order=[0,0], mode='nearest')
      dI1dx=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[0,1], mode='nearest')
      dI1dy=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[1,0], mode='nearest')
      dI2dx=ndimage.filters.gaussian_filter(frame2, [pxksz, pxksz], order=[0,1], mode='nearest')
      dI2dy=ndimage.filters.gaussian_filter(frame2, [pxksz, pxksz], order=[1,0], mode='nearest')

   else:
      weight=(pxsz/res)**2

      intframe1=frame1
      intframe2=frame2
      intmask1=mask1
      intmask2=mask2
      smoothframe1=frame1
      smoothframe2=frame2
      dI1dx=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[0,1], mode='nearest')
      dI1dy=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[1,0], mode='nearest')
      dI2dx=ndimage.filters.gaussian_filter(frame2, [1, 1], order=[0,1], mode='nearest')
      dI2dy=ndimage.filters.gaussian_filter(frame2, [1, 1], order=[1,0], mode='nearest') 

   # Calculation of the relative orientation angles
   #tempphi0=np.arctan2(grad1[1]*grad2[0]-grad1[0]*grad2[1], grad1[0]*grad2[0]+grad1[1]*grad2[1]) 
   tempphi=np.arctan2(dI1dx*dI2dy-dI1dy*dI2dx, dI1dx*dI2dx+dI1dy*dI2dy)
   phi=np.arctan(np.tan(tempphi))

   # Excluding small gradients
   normGrad1=np.sqrt(dI1dx**2+dI1dy**2) 
   normGrad2=np.sqrt(dI2dx**2+dI2dy**2) 
   bad=np.logical_or(normGrad1 <= gradthres1, normGrad2 <= gradthres2).nonzero()
   phi[bad]=np.nan

   corrframe=phi#np.cos(2.*phi)

   # Excluding masked regions   
   if np.array_equal(np.shape(intframe1), np.shape(intmask1)):
      corrframe[(intmask1 == 0.).nonzero()]=np.nan
      if np.array_equal(np.shape(intframe2), np.shape(intmask2)):
         corrframe[(intmask2 == 0.).nonzero()]=np.nan
         good=np.logical_and(np.logical_and(np.isfinite(phi), intmask1 > 0), intmask2 > 0).nonzero()
      else:
         good=np.logical_and(np.isfinite(phi), intmask1 > 0).nonzero()
   else:
         good=np.isfinite(phi).nonzero()

   Zx, s_Zx, meanPhi = HOG_PRS(phi[good])

   wghts=0.*phi[good]+weight

   rvl=circ.descriptive.resultant_vector_length(2.*phi[good], w=wghts)
   can=circ.descriptive.mean(2.*phi[good], w=wghts)/2.
   pz, Z = circ.tests.rayleigh(2.*phi[good],  w=wghts)
   pv, V = circ.tests.vtest(2.*phi[good], 0., w=wghts)

   myV, s_myV, meanphi = HOG_PRS(2.*phi[good])

   am = HOG_AM(phi[good])
 
   circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am]

   return circstats, corrframe, smoothframe1, smoothframe2
   
# -------------------------------------------------------------------------------------------------------------------------------
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

   pxksz =ksz/pxsz
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
   Zx, s_Zx, meanPhi = HOG_PRS(phi[good])

   return Zx, corrframe, smoothframe1



