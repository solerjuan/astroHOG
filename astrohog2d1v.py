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
from scipy import ndimage

from nose.tools import assert_equal, assert_true

import matplotlib.pyplot as plt

import collections
import multiprocessing

from astrohog2d import *
from statests import *

from tqdm import tqdm

# ------------------------------------------------------------------------------------------------------------------
CorrMapPair = collections.namedtuple('CorrMapPair', [
   'map1','map2',
   'pos1','pos2', 
   'pxsz','ksz','res',
   'mask1','mask2',
   'gradthres1','gradthres2',
   'wd'
])


# ------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------
def HOGppvblocks(corrcube, nbx=7, nby=7, vlims=[0.,1.,0.,1.], weight=1.):
   # Uses the pre-calculated global HOG correlation to calculate the HOG correlation in block of the map
   #
   # INPUTS
   #
   # correcube -- output of HOGcorr_ppvcubes function containing relative orientation angles between gradients 
   #
   # OUTPUTS
   #

   sz=np.shape(corrcube)
   x=(np.arange(0,sz[2],1)/(sz[2]/nbx)).astype(int)
   y=(np.arange(0,sz[3],1)/(sz[3]/nby)).astype(int)
   xx, yy = np.meshgrid(x, y)
   #limsx=np.linspace(0,sz[2]-1,nbx+1,dtype=int)
   #limsy=np.linspace(0,sz[3]-1,nby+1,dtype=int)

   zblocks=np.zeros([sz[0],sz[1],nbx,nby]) 
   vblocks=np.zeros([sz[0],sz[1],nbx,nby])

   maxvblocks=np.zeros([nbx,nby])
   sigvblocks=np.zeros([nbx,nby])

   # Loop over blocks 
   print("Block averaging ==========================")
   for i in tqdm(range(0, nby)):
      for k in range(0, nbx):

         # Loop over velocity channels
         for vi in range(0, sz[0]):
            for vk in range(0, sz[1]):
               phiframe=corrcube[vi,vk,:,:]  
               goodpos=np.logical_and(yy==i,xx==k).nonzero()
               phi=np.ravel(phiframe[goodpos])     
               wghts=weight*np.ones_like(phi)
               good=np.isfinite(phi).nonzero()

               if (np.size(good) > 1):
                  output=HOG_PRS(2.*phi[good], weights=wghts[good])
                  zblocks[vi,vk,k,i]=output['Z']
                  vblocks[vi,vk,k,i]=output['Zx']
               else:
                  zblocks[vi,vk,k,i]=np.nan
                  vblocks[vi,vk,k,i]=np.nan 

         tempvblocks=vblocks[:,:,k,i]
         if (np.size(np.isfinite(tempvblocks).nonzero()) > 0):
            maxvblocks[i,k]=np.max(tempvblocks[np.isfinite(tempvblocks).nonzero()])
            sigvblocks[i,k]=np.std(tempvblocks[np.isfinite(tempvblocks).nonzero()])
         else:
            maxvblocks[i,k]=np.nan
            sigvblocks[i,k]=np.nan
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

   imaxb, jmaxb = (maxvblocks==np.nanmax(maxvblocks)).nonzero()

   # Output circular statistics for the block with the highest V
   circstats={'Z': zblocks[:,:,imaxb[0], jmaxb[0]], 
              'V': vblocks[:,:,imaxb[0], jmaxb[0]]}

   #return [limsx[imaxb[0]],limsx[imaxb[0]+1],limsy[jmaxb[0]],limsy[jmaxb[0]+1]], vblocks[:,:,imaxb[0], jmaxb[0]], maxvblocks
   #return circstats, maxvblocks, xx, yy
   #return vblocks, maxvblocks, xx, yy
   return vblocks, xx, yy

# ================================================================================================================
def HOGcorr_ppvcubes(cube1, cube2, z1min, z1max, z2min, z2max, pxsz=1., ksz=1., res=1., mask1=0, mask2=0, gradthres1=0., gradthres2=0., s_cube1=0., s_cube2=0., nruns=0, weights=None, verbose=True):
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
 
   # Circular statistic outputs of orientation between image gradients
   rplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_rplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   zplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_zplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   vplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_vplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   meanphiplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_meanphiplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 

   # Circular statistic outputs of directions between image gradients   
   rdplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_rdplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   zdplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_zdplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
   vdplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_vdplane=np.zeros([z1max+1-z1min, z2max+1-z2min]) 
   meanphidplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_meanphidplane=np.zeros([z1max+1-z1min, z2max+1-z2min])

   # Correlation statistics  
   pearplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_pearplane=np.zeros([z1max+1-z1min, z2max+1-z2min]);
   ccorplane=np.zeros([z1max+1-z1min, z2max+1-z2min]); s_ccorplane=np.zeros([z1max+1-z1min, z2max+1-z2min]);
   neleplane=np.zeros([z1max+1-z1min, z2max+1-z2min])
 
   # -----------------------------------------------------------------
   corrframe=np.zeros([sz1[1],sz1[2]]) 
   scube1=np.zeros(sz1)
   scube2=np.zeros(sz2)

   corrcube=np.zeros([z1max+1-z1min, z2max+1-z2min,sz1[1],sz1[2]])           
   corrframe_temp=np.zeros([sz1[1],sz1[2]]) 
   maskcube=np.zeros(sz1) 

   pbar = tqdm(total=(z1max-z1min)*(z2max-z2min))

   # Loop over channel pairs
   for i in tqdm(range(z1min, z1max+1)):
      for k in range(z2min, z2max+1):
         vprint('Channel '+str(i-z1min)+'/'+str(z1max-z1min)+' and '+str(k-z2min)+'/'+str(z2max-z2min))
         frame1=cube1[i,:,:]
         frame2=cube2[k,:,:]

         if np.array_equal(np.shape(cube1), np.shape(mask1)):
            if np.array_equal(np.shape(cube2), np.shape(mask2)):				
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], mask2=mask2[k,:,:], gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights, verbose=verbose)
            else:
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, mask1=mask1[i,:,:], gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights, verbose=verbose)
         else:
               circstats, corrframe, sframe1, sframe2 = HOGcorr_ima(frame1, frame2, pxsz=pxsz, ksz=ksz, res=res, gradthres1=gradthres1, gradthres2=gradthres2, s_ima1=s_cube1, s_ima2=s_cube2, nruns=nruns, weights=weights, verbose=verbose)

         rplane[i-z1min,k-z2min] =circstats['RVL']; s_rplane[i-z1min,k-z2min] =circstats['s_RVL']
         zplane[i-z1min,k-z2min] =circstats['Z'];   s_zplane[i-z1min,k-z2min] =circstats['s_Z']
         vplane[i-z1min,k-z2min] =circstats['V'];   s_vplane[i-z1min,k-z2min] =circstats['s_V']
         meanphiplane[i-z1min,k-z2min] =circstats['meanphi']; s_meanphiplane[i-z1min,k-z2min] =circstats['s_meanphi']

         rdplane[i-z1min,k-z2min] =circstats['RVLd']; s_rdplane[i-z1min,k-z2min] =circstats['s_RVLd']
         zdplane[i-z1min,k-z2min] =circstats['Zd'];   s_zdplane[i-z1min,k-z2min] =circstats['s_Zd']
         vdplane[i-z1min,k-z2min] =circstats['Vd'];   s_vdplane[i-z1min,k-z2min] =circstats['s_Vd']  
         meanphidplane[i-z1min,k-z2min] =circstats['meanphid']; s_meanphidplane[i-z1min,k-z2min] =circstats['s_meanphid']

         pearplane[i-z1min,k-z2min]=circstats['pearsonr'] 
         s_pearplane[i-z1min,k-z2min]=circstats['s_pearsonr'] 
         ccorplane[i-z1min,k-z2min]=circstats['crosscor']  
         s_ccorplane[i-z1min,k-z2min]=circstats['s_crosscor']

         neleplane[i-z1min,k-z2min]=circstats['ngood']
            
         corrcube[i-z1min,k-z2min,:,:]=corrframe     
  
         scube2[k,:,:]=sframe2
      
         pbar.update()

      scube1[i,:,:]=sframe1 

   pbar.close() 

   outcircstats={'RVL': rplane, 'Z': zplane, 'V': vplane, 'meanphi': meanphiplane, 
                 's_RVL': s_rplane, 's_Z': s_zplane, 's_V': s_vplane, 's_meanphi': s_meanphiplane,
                 'RVLd': rdplane, 'Zd': zdplane, 'Vd': vdplane, 'meanphid': meanphidplane,
                 's_RVLd': s_rdplane, 's_Zd': s_zdplane, 's_Vd': s_vdplane, 's_meanphid': s_meanphidplane,
                 'pearsonr': pearplane, 's_pearsonr': s_pearplane,
                 'crosscor': ccorplane, 's_crosscor': s_ccorplane,  
                 'ngood': neleplane}

   return outcircstats, corrcube, scube1, scube2    

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



