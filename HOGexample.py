
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from scipy import misc
import numpy as np

# ======================================================================================================================
def HOGexample(im0, im1, pxksz=1, prefix='HOGExample'):

   plt.imshow(im0, cmap='gray')
   #plt.show()
   plt.savefig(prefix+'_Image0.png', bbox_inches='tight')
   plt.close()

   plt.imshow(im1, cmap='gray')
   #plt.show()
   plt.savefig(prefix+'_Image1.png', bbox_inches='tight')
   plt.close()

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
   grad1[0][0:2*pxksz+1,:]=np.nan;               grad1[1][0:2*pxksz+1,:]=np.nan
   grad1[0][sz1[0]-2*pxksz-2:sz1[0]-1,:]=np.nan; grad1[1][sz1[0]-2*pxksz-2:sz1[0]-1,:]=np.nan
   grad1[0][:,0:2*pxksz+1]=np.nan;               grad1[1][:,0:2*pxksz+1]=np.nan
   grad1[0][:,sz1[1]-2*pxksz-2:sz1[1]-1]=np.nan; grad1[1][:,sz1[1]-2*pxksz-2:sz1[1]-1]=np.nan
   normgrad1=np.sqrt(grad1[0]*grad1[0]+grad1[1]*grad1[1])

   # -------------------------------------------------------------------------------------------------------------------
   sz=np.shape(smap0)
   pitch=int(sz0[0]/20)
   X, Y = np.meshgrid(np.arange(2*pxksz, sz[1]-2*pxksz, pitch), np.arange(2*pxksz, sz[0]-2*pxksz, pitch))
   U0=(grad0[1]/normgrad0)[Y,X]
   V0=(grad0[0]/normgrad0)[Y,X]
 
   plt.imshow(normgrad0, cmap='gray')
   arrows=plt.quiver(X, Y, U0, V0, units='width', color='magenta', pivot='tail')
   plt.scatter(X, Y, color='magenta', s=0.25)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrows0.png', bbox_inches='tight')
   plt.close()

   U1=(grad1[1]/normgrad1)[Y,X]
   V1=(grad1[0]/normgrad1)[Y,X]

   plt.imshow(normgrad1, cmap='gray')
   arrows=plt.quiver(X, Y, U1, V1, units='width', color='cyan', pivot='tail')
   plt.scatter(X, Y, color='cyan', s=0.25)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrows1.png', bbox_inches='tight')
   plt.close()

   # --------------------------------------------------------------------------------------
   plt.imshow(normgrad0, cmap='gray', interpolation='none')
   arrows=plt.quiver(X, Y, U0, V0, units='width', color='magenta', pivot='tail')
   arrows=plt.quiver(X, Y, U1, V1, units='width', color='cyan', pivot='tail')
   plt.scatter(X, Y, color='magenta', s=0.25)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_GradImageWithArrowsBoth.png', bbox_inches='tight')
   plt.close()

   # --------------------------------------------------------------------------------------
   tempphi01=np.arctan2(grad0[0]*grad1[1]-grad0[1]*grad1[0], grad0[0]*grad1[0]+grad0[1]*grad1[1])
   phi01=np.arctan(np.tan(tempphi01))
   
   # Excluding small gradients
   gradthres=1. #1*np.mean([normgrad0,normgrad2])
   bad=np.logical_or(normgrad0 <= gradthres, normgrad1 <= gradthres).nonzero()
   phi01[bad]=np.nan

   plt.figure()
   ax=plt.gca()
   im=ax.imshow(np.abs(phi01)*180.0/np.pi, cmap='PiYG')
   divider=make_axes_locatable(ax)
   cax=divider.append_axes("right", size="5%", pad=0.05)
   #fig.colorbar(im, fraction=0.046, pad=0.04)
   plt.colorbar(im, cax=cax)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOGangles.png', bbox_inches='tight')
   plt.close()

   corrframe=np.cos(2.*phi01)

   im=plt.imshow(corrframe, cmap='seismic')
   plt.colorbar(im, fraction=0.046, pad=0.04)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOGcorr.png', bbox_inches='tight')
   plt.close()

   Zx01=np.sum(np.cos(2.*phi01[np.isfinite(phi01).nonzero()]))/np.sqrt(np.size(phi01[np.isfinite(phi01).nonzero()])/2.)
   temp=np.sum(np.cos(2.*phi01[np.isfinite(phi01).nonzero()])**2)
   s_Zx01=np.sqrt((2.*temp-Zx01*Zx01)/np.size(2.*phi01[np.isfinite(phi01).nonzero()]))
   print(str(Zx01)+'+/-'+str(s_Zx01))

   n, bins, patches = plt.hist(phi01[np.isfinite(phi01).nonzero()]*180.0/np.pi, 50, facecolor='green', alpha=0.5) #normed=1,
   plt.xlabel('Angle')
   plt.ylabel('Histogram frequency')
   plt.grid(True)
   #plt.show()
   plt.savefig(prefix+'_ksz'+str(pxksz)+'_HOG02.png', bbox_inches='tight')
   plt.close()

   #import pdb; pdb.set_trace() 


# ==========================================================================================================
im0=misc.imread('data/HOGExample001.png')
im1=misc.imread('data/HOGExample002.png')
HOGexample(im0[:,:,0], im1[:,:,0], pxksz=2, prefix='Examples/HOGExampleUncorr')
im2=misc.imread('data/HOGExample003.png')
HOGexample(im0[:,:,0], im2[:,:,0], pxksz=2, prefix='Examples/HOGExampleCorr')
#HOGexample(pxksz=3)
#HOGexample(pxksz=5)
#HOGexample(pxksz=9)


