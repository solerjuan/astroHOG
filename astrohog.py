# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from congrid import *
from scipy import ndimage

import pycircstat as circ
from nose.tools import assert_equal, assert_true

import matplotlib.pyplot as plt

import collections
import multiprocessing

CorrMapPair = collections.namedtuple('CorrMapPair', [
   'map1','map2',
   'pos1','pos2', 
   'pxsz','ksz','res',
   'mask1','mask2',
   'gradthres1','gradthres2',
   'wd'
])

def process_item(item):
   print('Process Item',item.pos1,item.pos2)
   circstats, corrframe, sframe1, sframe2 = HOGcorr_frame(item.map1, item.map2, pxsz=item.pxsz, ksz=item.ksz, res=item.res, mask1=item.mask1, mask2=item.mask2, gradthres1=item.gradthres1, gradthres2=item.gradthres2, wd=item.wd)
  
   return {
      'circstats': circstats,
      'corrframe': corrframe, 
      'sframe1': sframe1,
      'sframe2': sframe2, 
      'pos1': item.pos1,
      'pos2': item.pos2
   }

# ------------------------------------------------------------------------------------------------------------------------
def HOG_PRS(phi):
   # Calculates the projected Rayleigh statistic of the distributions of angles phi.
   #
   # INPUTS
   # phi      - angles between -pi/2 and pi/2
   #
   # OUTPUTS
   # Zx       - value of the projected Rayleigh statistic   
   # s_Zx     - 
   # meanPhi  -

   angles=phi #2.*phi

   Zx=np.sum(np.cos(angles))/np.sqrt(np.size(angles)/2.)	
   temp=np.sum(np.cos(angles)*np.cos(angles))
   s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))
   
   Zy=np.sum(np.sin(angles))/np.sqrt(np.size(angles)/2.)
   temp=np.sum(np.sin(angles)*np.sin(angles))
   s_Zx=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))

   meanPhi=0.5*np.arctan2(Zy, Zx)

   return Zx, s_Zx, meanPhi  

# ------------------------------------------------------------------------------------------------------------------------------
def HOG_AM(phi):
   # Calculate the alignment measure as introduced in Lazarian2007

   angles=phi
  
   ami=2.*np.cos(phi)-1. 
   am=np.mean(ami)

   return am

# -------------------------------------------------------------------------------------------------------------------------------
def HOGvotes_simple(phi):
    # Calculates the correlation   
    #
    # INPUTS
    #
    # OUTPUTS
    #
    #

    sz=np.shape(phi)
    corrframe=np.zeros(sz)	
    #paraThres=20.*np.pi/180.
    #condPara=np.logical_and(np.isfinite(phi), np.logical_or(phi < paraThres, phi > np.pi-paraThres)).nonzero() 
    #corrframe[condPara]=1.
    corrframe=np.cos(phi)	 	
    corrframe[np.isnan(phi).nonzero()]=0. #np.nan
    
    Zx, s_Zx, meanPhi = HOG_PRS(phi[np.isfinite(phi).nonzero()])	

    return Zx, corrframe


# -------------------------------------------------------------------------------------------------------------------------------
def HOGvotes_blocks(phi, wd=3):
    # Calculates the correlation   
    #
    # INPUTS
    #
    # OUTPUTS
    #
    #

   sz=np.shape(phi)
   corrframe=np.zeros(sz)

   for i in range(0, sz[0]):
      for k in range(0, sz[1]):
         if (i<wd):
            if (k<wd):
               temp=phi[0:i+wd,0:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            elif (k>sz[1]-1-wd):
               temp=phi[0:i+wd,k-wd:sz[1]-1]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            else:
               temp=phi[0:i+wd,k-wd:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
         elif (i>sz[0]-1-wd):
            if (k<wd):
               temp=phi[i-wd:sz[1]-1,0:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            elif (k>sz[1]-1-wd):
               temp=phi[i-wd:sz[0]-1,k-wd:sz[1]-1]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            else:
               temp=phi[i-wd:sz[0]-1,k-wd:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
         elif (k<wd):
            if (i<wd):
               temp=phi[0:i+wd,0:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            elif (i>sz[0]-1-wd):
               temp=phi[i-wd:sz[0]-1,0:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            else:
               temp=phi[i-wd:i+wd,0:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])	
         elif (k>sz[1]-1-wd):
            if (i<wd):
               temp=phi[0:i+wd,k-wd:sz[1]-1]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            elif (i>sz[0]-1-wd):
               temp=phi[i-wd:sz[0]-1,k-wd:sz[1]-1]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
            else:
               temp=phi[i-wd:i+wd,k-wd:sz[1]-1]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
         else:
            temp=phi[i-wd:i+wd,k-wd:k+wd]; corrframe[i,k]=np.mean(temp[np.isfinite(temp).nonzero()])
	
   corrframe[np.isnan(phi).nonzero()]=0.

   nPara=np.size(corrframe[(corrframe>0.).nonzero()])
   nGood=np.size(phi[np.isfinite(phi).nonzero()])

   hogcorr=nPara/float(nGood)

   return hogcorr, corrframe


# -------------------------------------------------------------------------------------------------------------------------------
def HOGcorr_frame(frame1, frame2, gradthres1=0., gradthres2=0., pxsz=1., ksz=1., res=1., mask1=0, mask2=0, wd=1, allow_huge=False, regrid=False):
   # Calculates the spatial correlation between frame1 and frame2 using the HOG methods
   #
   # INPUTS
   # frame1 -
   # frame2 -
   #
   # OUTPUTS
   # hogcorr -   
   # corrframe -

   sf=3. #Number of pixels per kernel FWHM	
     
   pxksz =ksz/pxsz 
   pxres =res/pxsz

   sz1=np.shape(frame1)  	

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
      #convolve_fft(intframe1, Gaussian2DKernel(pxksz), allow_huge=allow_huge)
      smoothframe2=ndimage.filters.gaussian_filter(frame2, [pxksz, pxksz], order=[0,0], mode='nearest') 
      #convolve_fft(intframe2, Gaussian2DKernel(pxksz), allow_huge=allow_huge)
      #grad1=np.gradient(smoothframe1)
      #grad2=np.gradient(smoothframe2)
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
      #grad1=np.gradient(intframe1)
      #grad2=np.gradient(intframe2)
      dI1dx=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[0,1], mode='nearest')
      dI1dy=ndimage.filters.gaussian_filter(frame1, [1, 1], order=[1,0], mode='nearest') 
      dI2dx=ndimage.filters.gaussian_filter(frame2, [1, 1], order=[0,1], mode='nearest')
      dI2dy=ndimage.filters.gaussian_filter(frame2, [1, 1], order=[1,0], mode='nearest')
    
   # Calculation of the relative orientation angles
   #tempphi0=np.arctan2(grad1[1]*grad2[0]-grad1[0]*grad2[1], grad1[0]*grad2[0]+grad1[1]*grad2[1]) 
   tempphi=np.arctan2(dI1dx*dI2dy-dI1dy*dI2dx, dI1dx*dI2dx+dI1dy*dI2dy)
   phi=np.arctan(np.tan(tempphi))

   # Excluding small gradients
   normGrad1=np.sqrt(dI1dx*dI1dx+dI1dy*dI1dy) #np.sqrt(grad1[1]**2+grad1[0]**2)
   normGrad2=np.sqrt(dI2dx*dI2dx+dI2dy*dI2dy) #np.sqrt(grad2[1]**2+grad2[0]**2)
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

   #if (wd > 1):
   #   hogcorr, corrframe =HOGvotes_blocks(phi, wd=wd)
   #else:
   #   hogcorr, corrframe =HOGvotes_simple(phi)
  
   circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am]

   return circstats, corrframe, smoothframe1, smoothframe2
   #return Zx, corrframe, smoothframe1


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
      #smoothframe1=convolve_fft(intframe1, Gaussian2DKernel(pxksz), allow_huge=allow_huge)
      smoothframe1=ndimage.filters.gaussian_filter(frame1, [pxksz, pxksz], order=[0,0], mode='nearest')
      #grad1=np.gradient(smoothframe1)
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
   #tempphi=np.arctan2(grad1[1]*intvecy-grad1[0]*intvecx, grad1[1]*intvecx+grad1[0]*intvecy)
   tempphi=np.arctan2(dI1dx*intvecy-dI1dy*intvecx, dI1dx*intvecx+dI1dy*intvecy)
   tempphi[bad]=np.nan
   phi=np.arctan(np.tan(tempphi))
	
   #if np.array_equal(np.shape(frame1), np.shape(mask1)):
   #   if np.array_equal(np.shape(normVec), np.shape(mask2)):
   #      phi[np.logical_or(mask1==0, mask2==0).nonzero()]=np.nan
   #	 good=np.logical_and(mask1 > 0., mask2 > 0.).nonzero()
   #   else:
   #      phi[(mask1==0).nonzero()]=np.nan
   #      good=(mask1 > 0.).nonzero()
   #else:
   #   good=np.isfinite(phi).nonzero()

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

   #if (wd > 1):
   #   hogcorr, corrframe =HOGvotes_blocks(phi, wd=wd)
   #else:
   #   hogcorr, corrframe =HOGvotes_simple(phi)

   #plt.imshow(phi, origin='lower')
   #plt.colorbar()
   #plt.show()
   #import pdb; pdb.set_trace() 

   return Zx, corrframe, smoothframe1


# ================================================================================================================
def HOGcorr_cube(cube1, cube2, z1min, z1max, z2min, z2max, pxsz=1., ksz=1., res=1., mask1=0, mask2=0, wd=1, gradthres1=0., gradthres2=0., regrid=False, allow_huge=False, multipro=False):

   # Calculates the correlation   
   #
   # INPUTS
   #
   # OUTPUTS
   #

   print('Computing HOG correlation')
   print(z1max-z1min+1,z2max-z2min+1) 

   sf=3. #Number of pixels per kernel FWHM
   pxksz =ksz/pxsz
   pxres =res/pxsz
   sz1=np.shape(cube1)
   sz2=np.shape(cube2)

   rplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   zplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   vplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   pzplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   pvplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 

   myvplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   mys_vplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   meanphiplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   amplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 

   corrplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   corrframe=np.zeros([sz1[1],sz1[2]]) 
   scube1=np.zeros(sz1)
   scube2=np.zeros(sz2)

   corrcube=np.zeros([z1max+1-z1min, z2max+1-z2min,sz1[1],sz1[2]])           
   corrframe_temp=np.zeros([sz1[1],sz1[2]]) 
   maskcube=np.zeros(sz1) 

   if (multipro):

      corrmappairs=(CorrMapPair(map1=np.zeros([sz1[1],sz1[2]]), map2=np.zeros([sz1[1],sz1[2]]), pos1=-9, pos2=-9, pxsz=pxsz, ksz=ksz, res=res, mask1=np.zeros([sz1[1],sz1[2]]), mask2=np.zeros([sz1[1],sz1[2]]), gradthres1=gradthres1, gradthres2=gradthres2, wd=wd),)
      count=0
   
      for i in range(z1min, z1max+1):
         for k in range(z2min, z2max+1):
            oldcorrmappairs=corrmappairs

            tempcorrmapair=CorrMapPair(map1=cube1[i,:,:], map2=cube2[k,:,:], pos1=i-z1min, pos2=k-z2min, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], mask2=mask2[k,:,:], gradthres1=gradthres1, gradthres2=gradthres2, wd=wd)   

            corrmappairs=oldcorrmappairs+(tempcorrmapair,) 

            del oldcorrmappairs
            del tempcorrmapair
 
            count+=1

      pool = multiprocessing.Pool()
      result = pool.map(process_item, corrmappairs) 

      for i in range(1, count):
         rplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][0]
         zplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][1]
         vplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][2]
         pzplane[result[i]['pos1'],result[i]['pos2']]=result[i]['circstats'][3]  
         pvplane[result[i]['pos1'],result[i]['pos2']]=result[i]['circstats'][4]
   
         myvplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][5] 
         mys_vplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][6]
         meanphiplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][7]
         amplane[result[i]['pos1'],result[i]['pos2']] =result[i]['circstats'][8]

         corrcube[result[i]['pos1'],result[i]['pos2'],:,:]=result[i]['corrframe']
         scube1[result[i]['pos1']+z1min,:,:]=result[i]['sframe1']
         scube2[result[i]['pos2']+z2min,:,:]=result[i]['sframe2']

   else:

      for i in range(z1min, z1max+1):
         for k in range(z2min, z2max+1):  
            print(i-z1min,k-z2min)
            frame1=cube1[i,:,:]
            frame2=cube2[k,:,:]
            if np.array_equal(np.shape(cube1), np.shape(mask1)):
               if np.array_equal(np.shape(cube2), np.shape(mask2)):				
                  circstats, corrframe, sframe1, sframe2 = HOGcorr_frame(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], mask2=mask2[k,:,:], gradthres1=gradthres1, gradthres2=gradthres2, wd=wd, regrid=regrid, allow_huge=allow_huge)
               else:
                  circstats, corrframe, sframe1, sframe2 = HOGcorr_frame(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], gradthres1=gradthres1, gradthres2=gradthres2, wd=wd, regrid=regrid, allow_huge=allow_huge)
            else:
               circstats, corrframe, sframe1, sframe2 = HOGcorr_frame(frame1, frame2, ksz=ksz, gradthres1=gradthres1, gradthres2=gradthres2, wd=wd, allow_huge=allow_huge)

            rplane[i-z1min,k-z2min]=circstats[0]
            zplane[i-z1min,k-z2min]=circstats[1]
            vplane[i-z1min,k-z2min]=circstats[2]
            pzplane[i-z1min,k-z2min]=circstats[3]
            pvplane[i-z1min,k-z2min]=circstats[4]
            
            myvplane[i-z1min,k-z2min]=circstats[5]
            mys_vplane[i-z1min,k-z2min]=circstats[6]
            meanphiplane[i-z1min,k-z2min]=circstats[7]
            amplane[i-z1min,k-z2min]=circstats[8]  
              
            corrcube[i-z1min,k-z2min,:,:]=corrframe
            scube2[k,:,:]=sframe2

         scube1[i,:,:]=sframe1 

   return [rplane,zplane,vplane,pzplane,pvplane,myvplane,mys_vplane,meanphiplane,amplane], corrcube, scube1, scube2


# ================================================================================================================
def HOGcorr_cubeandpol(cube1, ex, ey, z1min, z1max, pxsz=1., ksz=1., res=1., mask1=0, mask2=0, wd=1, rotatepol=False, regrid=False, allow_huge=False):
   # Calculates the correlation   
   #
   # INPUTS
   #
   # OUTPUTS
   #
   #

   print('Computing HOG correlation')
   print(z1max-z1min)

   sf=3. #Number of pixels per kernel FWHM      
   pxksz =ksz/pxsz
   pxres =res/pxsz
   sz1=np.shape(cube1)
   sz2=np.shape(ex)

   if(rotatepol):
      xvec= ey
      yvec=-ex
   else:	
      xvec= ex
      yvec= ey
   normVec=np.sqrt(xvec*xvec+yvec*yvec)

   corrvec=0.*np.arange(z1min,z1max+1)
   corrframe=np.zeros([sz1[1],sz1[2]])	
   corrcube=np.zeros(sz1)
   scube=np.zeros(sz1)

   for i in range(z1min, z1max+1):
      print(i-z1min)
      if np.array_equal(np.shape(cube1), np.shape(mask1)):
         if np.array_equal(np.shape(normVec), np.shape(mask2)):                
            corr, corrframe, sframe = HOGcorr_frameandvec(cube1[i,:,:], xvec, yvec, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], mask2=mask2, wd=wd, regrid=regrid)
         else:
            corr, corrframe, sframe = HOGcorr_frameandvec(cube1[i,:,:], xvec, yvec, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], wd=wd, regrid=regrid)
      else:
         corr, corrframe, sframe = HOGcorr_frameandvec(cube1[i,:,:], xvec, yvec, pxsz=pxsz, ksz=ksz, res=res, wd=wd, regrid=regrid)
      corrvec[i-z1min]=corr
      #corrcube[i-z1min]=corrframe
      corrcube[i,:,:]=corrframe
      scube[i,:,:]=sframe

   return corrvec, corrcube, scube



