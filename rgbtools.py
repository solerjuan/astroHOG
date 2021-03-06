# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler

import sys
import numpy as np
from astropy.io import fits
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

#sys.path.append('/disk2/soler/PYTHON/astroHOG/')
#from astrohog import *

from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from astropy.wcs import WCS
from reproject import reproject_interp

import imageio

SMALLER_SIZE=6

from matplotlib.colors import LinearSegmentedColormap

cdict1 = {'red':   ((0.00, 0.0, 0.0),
                    (0.01, 0.0, 0.0),
                    (0.02, 0.0, 0.0),
                    (0.03, 0.0, 0.0),
                    (0.04, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),

         'green':  ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

         'blue':   ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
        }

COcolort=LinearSegmentedColormap('COcmap', cdict1)
plt.register_cmap(cmap=COcolort)

cdict2 = {'red':   ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

         'green':  ((0.0, 0.00, 0.0),
                    (0.0, 0.01, 0.0),
                    (0.0, 0.02, 0.0),
                    (0.0, 0.03, 0.0),
                    (0.0, 0.04, 0.0),
                    (1.0, 1.00, 1.0)),

         'blue':   ((0.0, 0.00, 0.0),
                    (0.0, 0.01, 0.0),
                    (0.0, 0.02, 0.0),
                    (0.0, 0.03, 0.0),
                    (0.0, 0.04, 0.0),
                    (1.0, 1.00, 1.0))
        }

HIcolort=LinearSegmentedColormap('COcmap', cdict2)
plt.register_cmap(cmap=HIcolort)

# -----------------------------------------------------------------------------------------------------------
def tealct():
   
   return HIcolort

# -----------------------------------------------------------------------------------------------------------
def redct():
   
   return COcolort

# -----------------------------------------------------------------------------------------------------------
def rgbcube(cube, zmin, zmax, autoscale=False, minref=None, maxref=None, ksz=1, EquiBins=True, minauto=0.2, maxauto=0.975):

   nbins=1000
   sz=np.shape(cube)
   cube[np.isnan(cube).nonzero()]=0.

   rgb=np.zeros([sz[1],sz[2],3])

   channels=zmax-zmin+1
   indexes=np.arange(zmin,zmax)
   pitch=int(channels/3.)

   meanI=cube[zmin:zmax].mean(axis=(1,2))
   cumsumI=np.cumsum(meanI)
   binwd=np.max(cumsumI)/3.

   # ------------------------------------------------------------------------------------
   firstb=np.max((cumsumI < binwd).nonzero())
   if (EquiBins):
      tempcube=cube[zmin:zmin+firstb-1,:,:]
   else:
      tempcube=cube[zmin:zmin+pitch-1,:,:]
   tempcube[np.isnan(tempcube).nonzero()]=0.
   if (minref):
      tempcube[(tempcube<minref).nonzero()]=0.
   tempmap=tempcube.mean(axis=0)
    
   hist, bin_edges = np.histogram(tempmap, density=True, bins=nbins)
   bin_centres=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
   chist=np.cumsum(hist)

   cond=(chist < minauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      mini=np.min(bin_centres)
   else:
      mini=np.max(bin_edges[cond])
   cond=(chist > maxauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      maxi=np.max(bin_centres)
   else: 
      maxi=np.min(bin_edges[cond])
 
   if(autoscale): 
      tempmap[(tempmap < mini).nonzero()]=mini
      tempmap[(tempmap > maxi).nonzero()]=maxi

   red=(tempmap-np.min(tempmap))/(np.max(tempmap)-np.min(tempmap))

   # ------------------------------------------------------------------------------------
   secondb=np.max((cumsumI < 2.*binwd).nonzero())
   if (EquiBins):
      tempcube=cube[zmin+firstb:zmin+secondb,:,:]
   else:
      tempcube=cube[zmin+pitch:zmin+2*pitch-1,:,:]
   tempcube[np.isnan(tempcube).nonzero()]=0.
   if (minref):
      tempcube[(tempcube<minref).nonzero()]=0.
   tempmap=tempcube.mean(axis=0)

   hist, bin_edges = np.histogram(tempmap, density=True, bins=nbins)
   bin_centres=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
   chist=np.cumsum(hist)  

   cond=(chist < minauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      mini=np.min(bin_centres)
   else:
      mini=np.max(bin_edges[cond])
   cond=(chist > maxauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      maxi=np.max(bin_centres)
   else:
      maxi=np.min(bin_edges[cond])

   if(autoscale):
      tempmap[(tempmap < mini).nonzero()]=mini
      tempmap[(tempmap > maxi).nonzero()]=maxi

   green=(tempmap-np.min(tempmap))/(np.max(tempmap)-np.min(tempmap))

   # ------------------------------------------------------------------------------------ 
   if (EquiBins):
      tempcube=cube[zmin+secondb+1:zmax,:,:]
   else:
      tempcube=cube[zmin+2*pitch:zmax,:,:]
   tempcube[np.isnan(tempcube).nonzero()]=0.
   if (minref):
      tempcube[(tempcube<minref).nonzero()]=0.
   tempmap=tempcube.mean(axis=0)

   hist, bin_edges = np.histogram(tempmap, density=True, bins=nbins)
   bin_centres=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
   chist=np.cumsum(hist)

   cond=(chist < minauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      mini=np.min(bin_centres)
   else:
      mini=np.max(bin_edges[cond])
   cond=(chist > maxauto*chist[np.size(chist)-1]).nonzero()
   if (np.size(cond)==0):
      maxi=np.max(bin_centres)
   else:
      maxi=np.min(bin_edges[cond])

   if(autoscale):
      tempmap[(tempmap < mini).nonzero()]=mini
      tempmap[(tempmap > maxi).nonzero()]=maxi
   
   blue=(tempmap-np.min(tempmap))/(np.max(tempmap)-np.min(tempmap))

   rgb[:,:,0]=red
   rgb[:,:,1]=green
   rgb[:,:,2]=blue

   return rgb;

# -----------------------------------------------------------------------------------------------------------
def rgbmovie(cube, zmin, zmax, logscale=False, minref=0., maxref=0.45, ksz=1, group=2, prefix='frame', hdr=0, duration=0.5):

   sz=np.shape(cube)
   rgb=np.zeros([sz[1],sz[2],3])
   k=0

   images=[]

   for i in range(zmin, zmax):

      tempmap=cube[i-1-group/2:i-1+group/2,:,:].mean(axis=0)
      if(group==0):
         tempmap=cube[i-1,:,:]
      if(ksz > 1):
         inmap=convolve_fft(tempmap, Gaussian2DKernel(ksz))
      else:
         inmap=tempmap
      if(logscale):
         inmap=np.log10(np.copy(inmap))
      inmap[np.isinf(inmap).nonzero()]=minref 
      inmap[(inmap < minref).nonzero()]=minref
      inmap[(inmap > maxref).nonzero()]=maxref
      red=(inmap-np.min(inmap))/(np.max(inmap)-np.min(inmap))
 
      tempmap=cube[i-group/2:i+group/2,:,:].sum(axis=0)/float(group+1)
      if(group==0):
         tempmap=cube[i,:,:]
      if(ksz > 1):
         inmap=convolve_fft(tempmap, Gaussian2DKernel(ksz))
      else:
         inmap=tempmap
      if(logscale):
         inmap=np.log10(np.copy(inmap))
      inmap[np.isinf(inmap).nonzero()]=minref 
      inmap[(inmap < minref).nonzero()]=minref
      inmap[(inmap > maxref).nonzero()]=maxref
      green=(inmap-np.min(inmap))/(np.max(inmap)-np.min(inmap))

      tempmap=cube[i+1-group/2:i+1+group/2,:,:].sum(axis=0)/float(group+1)
      if(group==0):
         tempmap=cube[i+1,:,:]
      if(ksz > 1):
         inmap=convolve_fft(tempmap, Gaussian2DKernel(ksz))
      else:
         inmap=tempmap
      if(logscale):
         inmap=np.log10(np.copy(inmap))
      inmap[np.isinf(inmap).nonzero()]=minref 
      inmap[(inmap < minref).nonzero()]=minref
      inmap[(inmap > maxref).nonzero()]=maxref
      blue=(inmap-np.min(inmap))/(np.max(inmap)-np.min(inmap))

      rgb[:,:,0]=red
      rgb[:,:,1]=green
      rgb[:,:,2]=blue

      fig = plt.figure(figsize=(1.5, 3.0), dpi=300)
      plt.rc('font', size=SMALLER_SIZE)
      if(hdr):
         ax1=plt.subplot(1,1,1, projection=WCS(hdr)) 
         im=ax1.imshow(rgb, origin='lower', interpolation='none')
         ax1.coords.grid(color='white')
         ax1.coords['glon'].set_axislabel('Galactic Longitude')
         ax1.coords['glat'].set_axislabel('Galactic Latitude')
      else:
         ax1=plt.subplot(1,1,1)
         im=ax1.imshow(rgb, origin='lower', interpolation='none')
      rgb=rgbcube(cube1, zmin1, zmax1, minref=minrm1, EquiBins=False)
      ax1.set_title('Projected HI')
      #plt.show()  
      plt.savefig(prefix+'_'+str(k)+'.png', bbox_inches='tight')
      plt.close() 
 
      images.append(imageio.imread(prefix+'_'+str(k)+'.png'))

      k+=1

   imageio.mimsave(prefix+'.gif', images, duration=duration)   

      #import pdb; pdb.set_trace()

