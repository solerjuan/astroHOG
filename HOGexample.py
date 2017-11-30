
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from scipy import misc
import numpy as np

SMALLER_SIZE=5
BIGGER_SIZE=15

# ======================================================================================================================
def makeExampleImage():

   #im0=misc.imread('data/Haeckel_Asteridea.jpg')#[:,:,0]
   im0=np.transpose(misc.imread('data/Haeckel_Nudibranchia.png'))   

   sz=np.shape(im0)   

   red=im0
   green=np.roll(np.roll(im0, int(sz[0]*0.5), axis=0), int(sz[1]*0.5), axis=1)
   blue=np.roll(np.roll(im0, -int(sz[0]*0.2), axis=0), -int(sz[1]*0.2), axis=1)

   rgb=np.zeros([sz[0],sz[1],3])
   rgb[:,:,0]=red#255-red
   rgb[:,:,1]=green#255-green
   rgb[:,:,2]=blue#255-blue
 
   #plt.imshow(rgb, origin='lower')
   #plt.show()

   import scipy.misc
   scipy.misc.imsave('data/uniHaeckel_Nudibranchia.png',   im0)
   scipy.misc.imsave('data/multiHaeckel_Nudibranchia.png', rgb)

   #import pdb; pdb.set_trace()

# ======================================================================================================================
def HOGexample(im0, im1, pxksz=1, prefix='HOGExample'):

   plt.figure(figsize=(12., 5.))
   plt.imshow(im0, cmap='gray', interpolation='none')
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_Image0.png', bbox_inches='tight', dpi=300)
   plt.close()

   plt.figure(figsize=(12., 5.))
   plt.imshow(im1, interpolation='none') 
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_Image1.png', bbox_inches='tight', dpi=300)
   plt.close()

   #im0=im0.sum(axis=2)
   im1=im1.sum(axis=2) 
   #import pdb; pdb.set_trace()
   # -------------------------------------------------------------------------------------------------------------------
   if (pxksz > 1):
      smap0=convolve_fft(im0, Gaussian2DKernel(pxksz))
   else:
      smap0=im0

   grad0=np.gradient(smap0)
   sz0=np.shape(smap0)
   grad0[0][0:2*pxksz+1,:]=np.nan;                 grad0[1][0:2*pxksz+1,:]=np.nan;
   grad0[0][sz0[0]-2*pxksz-1:sz0[0]-1,:]=np.nan;   grad0[1][sz0[0]-2*pxksz-1:sz0[0]-1,:]=np.nan;
   grad0[0][:,0:2*pxksz+1]=np.nan;                 grad0[1][:,0:2*pxksz+1]=np.nan;
   grad0[0][:,sz0[1]-2*pxksz-1:sz0[1]-1]=np.nan;   grad0[1][:,sz0[1]-2*pxksz-1:sz0[1]-1]=np.nan;
   normgrad0=np.sqrt(grad0[0]*grad0[0]+grad0[1]*grad0[1])

   if (pxksz > 1):
      smap1=convolve_fft(im1, Gaussian2DKernel(pxksz))
   else:
      smap1=im1

   grad1=np.gradient(smap1)
   sz1=np.shape(smap1)
   grad1[0][0:2*(pxksz+1),:]=np.nan;               grad1[1][0:2*(pxksz+1),:]=np.nan
   grad1[0][sz1[0]-2*(pxksz+1):sz1[0]-1,:]=np.nan; grad1[1][sz1[0]-2*(pxksz+1):sz1[0]-1,:]=np.nan
   grad1[0][:,0:2*(pxksz+1)]=np.nan;               grad1[1][:,0:2*(pxksz+1)]=np.nan
   grad1[0][:,sz1[1]-2*(pxksz+1):sz1[1]-1]=np.nan; grad1[1][:,sz1[1]-2*(pxksz+1):sz1[1]-1]=np.nan
   normgrad1=np.sqrt(grad1[0]*grad1[0]+grad1[1]*grad1[1])

   # -------------------------------------------------------------------------------------------------------------------
   sz=np.shape(smap0)
   pitch=int(sz0[0]/25)
   X, Y = np.meshgrid(np.arange(2*pxksz, sz[1]-2*pxksz, pitch), np.arange(2*pxksz, sz[0]-2*pxksz, pitch))

   dnormgrad0=normgrad0
   dnormgrad0[(dnormgrad0 == 0.)]=1.
   U0=(grad0[1]/dnormgrad0)[Y,X]
   V0=(grad0[0]/dnormgrad0)[Y,X]
 
   plt.figure(figsize=(12., 5.))
   plt.imshow(normgrad0, cmap='gray_r', interpolation='none', clim=[1.,0.8*np.max(normgrad0[np.isfinite(normgrad0).nonzero()])])
   arrows=plt.quiver(X, Y, U0, V0, units='width', color='red', pivot='tail')
   plt.scatter(X, Y, color='red', s=0.2)
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrows0.png', bbox_inches='tight', dpi=300)
   plt.close()

   dnormgrad1=normgrad1
   dnormgrad1[(dnormgrad1 == 0.)]=1.
   U1=(grad1[1]/dnormgrad1)[Y,X]
   V1=(grad1[0]/dnormgrad1)[Y,X]

   plt.figure(figsize=(12., 5.))
   plt.imshow(normgrad1, cmap='gray_r', interpolation='none', clim=[1.,0.3*np.max(normgrad1[np.isfinite(normgrad1).nonzero()])])
   arrows=plt.quiver(X, Y, U1, V1, units='width', color='blue', pivot='tail')
   plt.scatter(X, Y, color='blue', s=0.2)
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrows1.png', bbox_inches='tight', dpi=300)
   plt.close()
   #import pdb; pdb.set_trace()
   # --------------------------------------------------------------------------------------
   plt.figure(figsize=(12., 5.))
   plt.imshow(normgrad0, cmap='gray_r', interpolation='none')
   arrows=plt.quiver(X, Y, U0, V0, units='width', color='magenta', pivot='tail')
   arrows=plt.quiver(X, Y, U1, V1, units='width', color='cyan', pivot='tail')
   plt.scatter(X, Y, color='magenta', s=0.25)
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrowsBoth.png', bbox_inches='tight', dpi=300)
   plt.close()

   # --------------------------------------------------------------------------------------
   tempphi01=np.arctan2(grad0[0]*grad1[1]-grad0[1]*grad1[0], grad0[0]*grad1[0]+grad0[1]*grad1[1])
   phi01=np.arctan(np.tan(tempphi01))
   
   # Excluding small gradients
   gradthres=0. #1*np.mean([normgrad0,normgrad2])
   bad=np.logical_or(normgrad0 <= gradthres, normgrad1 <= gradthres).nonzero()
   phi01[bad]=np.nan

   plt.figure(figsize=(12., 5.))
   im=plt.imshow(phi01*180.0/np.pi, cmap='RdGy_r', interpolation='none')
   plt.colorbar(im, orientation='horizontal', fraction=0.06, pad=0.025)
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOGangles.png', bbox_inches='tight', dpi=300)
   plt.close()

   corrframe=np.cos(2.*phi01)

   plt.figure(figsize=(12., 5.))
   im=plt.imshow(corrframe, cmap='seismic')
   plt.colorbar(im, orientation='horizontal', fraction=0.06, pad=0.025)
   plt.xticks([])
   plt.yticks([])
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOGcorr.png', bbox_inches='tight', dpi=300)
   plt.close()
   #import pdb; pdb.set_trace() 

   Zx01=np.sum(np.cos(2.*phi01[np.isfinite(phi01).nonzero()]))/np.sqrt(np.size(phi01[np.isfinite(phi01).nonzero()])/2.)
   temp=np.sum(np.cos(2.*phi01[np.isfinite(phi01).nonzero()])**2)
   s_Zx01=np.sqrt((2.*temp-Zx01*Zx01)/np.size(2.*phi01[np.isfinite(phi01).nonzero()]))
   print(str(Zx01)+'+/-'+str(s_Zx01))

   n, bins, patches = plt.hist(phi01[np.isfinite(phi01).nonzero()]*180.0/np.pi, 50, facecolor='red', alpha=0.5) #normed=1,
   plt.xlabel('Angle')
   plt.ylabel('Histogram frequency')
   plt.grid(True)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOG.png', bbox_inches='tight', dpi=300)
   plt.close()

   #import pdb; pdb.set_trace() 


# ==========================================================================================================
#makeExampleImage()
im0=misc.imread('data/uniHaeckel_Nudibranchia.png')
im1=misc.imread('data/multiHaeckel_Nudibranchia.png')
#import pdb; pdb.set_trace()
#HOGexample(im0, im1, pxksz=2, prefix='Examples/HOGExampleCorr')
HOGexample(im0, im1, pxksz=3, prefix='Examples/HOGExampleCorr')
#import pdb; pdb.set_trace()
#im2=misc.imread('data/HOGExample003.png')
#HOGexample(im0[:,:,0], im2[:,:,0], pxksz=2, prefix='Examples/HOGExampleCorr')
#HOGexample(pxksz=3)
#HOGexample(pxksz=5)
#HOGexample(pxksz=9)


