# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

def HOGvotes_simple(phi):

	paraThres=20.*np.pi/180.
	
	condPara=np.logical_and(np.isfinite(phi), np.logical_or(phi < paraThres, phi > np.pi-paraThres)).nonzero() 	
	#condPara=np.logical_and(np.isfinite(phi), np.logical_or(phi < paraThres, phi > 180.-paraThres)).nonzero() 
	nPara=np.size(phi[condPara])
	condGood=np.isfinite(phi).nonzero()
        nGood=np.size(phi[condGood])

	sz=np.shape(phi)
        corrframe=np.zeros(sz)
        corrframe[condPara]=1.

        hogcorr=nPara/float(nGood)
	return hogcorr, corrframe


def HOGvotes_blocks(phi, wd=3):

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


def HOGcorr_frame(frame1, frame2, gradthres=0., ksz=1, mask1=0, mask2=0, wd=1):

	sz1=np.shape(frame1)

	if (ksz > 1):
		grad1=np.gradient(convolve_fft(frame1, Gaussian2DKernel(ksz)))
		grad2=np.gradient(convolve_fft(frame2, Gaussian2DKernel(ksz)))	
	else:
		grad1=np.gradient(frame1)
		grad2=np.gradient(frame2)

	normGrad1=np.sqrt(grad1[1]**2+grad1[0]**2)
        normGrad2=np.sqrt(grad2[1]**2+grad2[0]**2)
	bad=np.logical_or(normGrad1 <= gradthres, normGrad2 <= gradthres).nonzero()

	#cphi=(grad1[1]*grad2[1]+grad1[0]*grad2[0])
	#sphi=(grad1[1]*grad2[0]-grad1[0]*grad2[1])
	#phi=np.arctan2(sphi, cphi)
	normGrad1[bad]=1.; normGrad2[bad]=1.; 	
	cosphi=(grad1[1]*grad2[1]+grad1[0]*grad2[0])/(normGrad1*normGrad2)
	phi=np.arccos(cosphi)

	phi[bad]=np.nan
	if np.array_equal(np.shape(frame1), np.shape(mask1)):
		if np.array_equal(np.shape(frame2), np.shape(mask2)):		
			phi[np.logical_or(mask1==0, mask2==0).nonzero()]=np.nan
		else:	
			phi[(mask1==0).nonzero()]=np.nan

	if (wd > 1):
		hogcorr, corrframe =HOGvotes_blocks(phi, wd=wd)
	else:
		hogcorr, corrframe =HOGvotes_simple(phi)

	#plt.imshow(phi, origin='lower')
        #plt.show()
	#import pdb; pdb.set_trace()

	return hogcorr, corrframe

def HOGcorr_cube(cube1, cube2, z1min, z1max, z2min, z2max, ksz=1, mask1=0, mask2=0, wd=1):

	sz1=np.shape(cube1)
	sz2=np.shape(cube2)

	corrplane=np.zeros([z1max-z1min, z2max-z2min])
	corrcube=np.zeros(sz1)

	for i in range(z1min, z1max):
                for j in range(z2min, z2max):
				if np.array_equal(np.shape(cube1), np.shape(mask1)):
					if np.array_equal(np.shape(cube2), np.shape(mask2)):				
						corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz, mask1=mask1[i,:,:], mask2=mask2[i,:,:], wd=wd)
					else:
						corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz, mask1=mask1[i,:,:], wd=wd)
				else:
					corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz, wd=wd)
				corrplane[i-z1min,j-z2min]=corr

	return corrplane


