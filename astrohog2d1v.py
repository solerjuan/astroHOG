#!/usr/bin/env python
#
# This file is part of AstroHOG
#
# CONTACT: soler[AT]mpia.de
# Copyright (C) 2013-2017 Juan Diego Soler
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

# --------------------------------------------------------------------------------------------------------------------------------
def BlockAverage(corrcube, nbx=7, nby=7, vlims=[0.,1.,0.,1.], weight=1.):

   sz=np.shape(corrcube)
   limsx=np.linspace(0,sz[2]-1,nbx+1,dtype=int)
   limsy=np.linspace(0,sz[3]-1,nby+1,dtype=int)
   vblocks=np.zeros([sz[0],sz[1],nbx,nby])

   maxvblocks=np.zeros([nbx,nby])
   sigvblocks=np.zeros([nbx,nby])

   for i in range(0, np.size(limsx)-1):
      for k in range(0, np.size(limsy)-1):

         for vi in range(0, sz[0]):
            for vk in range(0, sz[1]):

               phi = corrcube[vi,vk,limsx[i]:limsx[i+1],limsy[k]:limsy[k+1]]
               tempphi=phi.ravel()
               wghts=0.*tempphi[np.isfinite(tempphi).nonzero()]+weight
               pz, Zx = circ.tests.vtest(2.*tempphi[np.isfinite(tempphi).nonzero()],0.,w=wghts)
               vblocks[vi,vk,i,k] = Zx

         tempvblocks=vblocks[:,:,i,k]
         maxvblocks[i,k]=np.max(tempvblocks[np.isfinite(tempvblocks).nonzero()])
         sigvblocks[i,k]=np.std(tempvblocks[np.isfinite(tempvblocks).nonzero()])

   #if (np.logical_and(nbx==1,nby==1)):
   #   fig, ax = plt.subplots(figsize = (9.0,8.0))
   #   im=ax.imshow(vblocks[:,:,0,0], origin='lower', extent=vlims, vmin=0., vmax=np.max(maxvblocks), aspect='auto')
   #   ax.set_xlabel(r'$v_{CO}$ [km/s]')
   #   ax.set_ylabel(r'$v_{HI}$ [km/s]')
   #   cbl=plt.colorbar(im, ax=ax)
   #   cbl.ax.set_title(r'$V$')
   #   plt.show()
   #else:
   #   fig, axs = plt.subplots(nbx,nby,figsize = (9.0,8.0))
   #   fig.subplots_adjust(hspace=0.001, wspace=0.005)
   #   for i in range(0,nbx):
   #      for k in range(0, nby):
   #         im=axs[nby-1-i,k].imshow(vblocks[:,:,i,k], origin='lower', extent=vlims, vmin=0., vmax=np.max(maxvblocks), aspect='auto')
   #         if(np.logical_and(i==nby-1,k==0)):
   #            axs[i,k].set_xlabel(r'$v_{CO}$ [km/s]')
   #            axs[i,k].set_ylabel(r'$v_{HI}$ [km/s]')
   #   cbl=plt.colorbar(im, ax=axs.ravel().tolist())
   #   cbl.ax.set_title(r'$V$')
   #   plt.show()

   imaxb, jmaxb = (maxvblocks==np.max(maxvblocks)).nonzero()

   return [limsx[imaxb[0]],limsx[imaxb[0]+1],limsy[jmaxb[0]],limsy[jmaxb[0]+1]], vblocks[:,:,imaxb[0], jmaxb[0]], maxvblocks


# ================================================================================================================
def HOGcorr_ppvcubes(cube1, cube2, z1min, z1max, z2min, z2max, pxsz=1., ksz=1., res=1., mask1=0, mask2=0, gradthres1=0., gradthres2=0., s_cube1=0., s_cube2=0., nruns=0, weights=None):
   # Calculates the HOG correlation between PPV cubes   
   #
   # INPUTS
   #
   # OUTPUTS
   #

   print('Computing HOG correlation')
   print(z1max-z1min+1,z2max-z2min+1) 
  
   sz1=np.shape(cube1)
   sz2=np.shape(cube2)

   rplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   zplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   vplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   amplane=np.zeros([z1max+1-z1min, z2max+1-z2min])   

   pearplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   neleplane=np.zeros([z1max+1-z1min, z2max+1-z2min])

   rplane0=np.zeros([z1max+1-z1min, z2max+1-z2min])
   vplane0=np.zeros([z1max+1-z1min, z2max+1-z2min]) 

   s_rplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 
   s_zplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 
   s_vplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   s_amplane=np.zeros([z1max+1-z1min, z2max+1-z2min])

   corrframe=np.zeros([sz1[1],sz1[2]]) 
   scube1=np.zeros(sz1)
   scube2=np.zeros(sz2)

   corrcube=np.zeros([z1max+1-z1min, z2max+1-z2min,sz1[1],sz1[2]])           
   corrframe_temp=np.zeros([sz1[1],sz1[2]]) 
   maskcube=np.zeros(sz1) 

   pbar = tqdm(total=(z1max-z1min)*(z2max-z2min))

   for i in range(z1min, z1max+1):
      for k in range(z2min, z2max+1):  
         print('Channel '+str(i-z1min)+'/'+str(z1max-z1min)+' and '+str(k-z2min)+'/'+str(z2max-z2min))
         frame1=cube1[i,:,:]
         frame2=cube2[k,:,:]
         if np.array_equal(np.shape(cube1), np.shape(mask1)):
            if np.array_equal(np.shape(cube2), np.shape(mask2)):				
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], mask2=mask2[k,:,:], gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights)
            else:
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights)
         else:
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights)

         # circstats=[meanr, meanz, meanv, s_r, s_z, s_v, outr, outv, am, s_am]
         rplane[i-z1min,k-z2min] =circstats[0]
         zplane[i-z1min,k-z2min] =circstats[1]
         vplane[i-z1min,k-z2min] =circstats[2]
         amplane[i-z1min,k-z2min]=circstats[8]

         pearplane[i-z1min,k-z2min]=circstats[10] 
         neleplane[i-z1min,k-z2min]=circstats[11]

         rplane0[i-z1min,k-z2min] =circstats[6]
         vplane0[i-z1min,k-z2min] =circstats[7]
            
         s_rplane[i-z1min,k-z2min] =circstats[3]
         s_rplane[i-z1min,k-z2min] =circstats[4]
         s_vplane[i-z1min,k-z2min] =circstats[5]
         s_amplane[i-z1min,k-z2min]=circstats[9]
         
         corrcube[i-z1min,k-z2min,:,:]=corrframe     
  
         scube2[k,:,:]=sframe2
      
         pbar.update()

      scube1[i,:,:]=sframe1 

   pbar.close() 

   return [rplane,zplane,vplane,s_rplane,s_zplane,s_vplane,rplane0,vplane0,amplane,s_amplane,pearplane,neleplane], corrcube, scube1, scube2


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



