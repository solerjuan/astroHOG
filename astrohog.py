# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

def HOGcorr_frame(frame1, frame2, gradthres=0., ksz=1, mask1=0, mask2=0):

	sz1=np.shape(frame1)

	if (ksz > 1):
		grad1=np.gradient(convolve_fft(frame1, Gaussian2DKernel(ksz)))
		grad2=np.gradient(convolve_fft(frame2, Gaussian2DKernel(ksz)))	
	else:
		grad1=np.gradient(frame1)
		grad2=np.gradient(frame2)

	normGrad1=np.sqrt(grad1[1]**2+grad1[0]**2)
        normGrad2=np.sqrt(grad2[1]**2+grad2[0]**2)

	cphi=(grad1[1]*grad2[1]+grad1[0]*grad2[0])
	sphi=(grad1[1]*grad2[0]-grad1[0]*grad2[1])
	phi=np.arctan2(sphi, cphi)

	bad=np.logical_and(normGrad1 <= gradthres, normGrad2 <= gradthres).nonzero()
	phi[bad]=np.nan
	if np.array_equal(np.shape(frame1), np.shape(mask1)):
		if np.array_equal(np.shape(frame2), np.shape(mask2)):		
			phi[np.logical_or(mask1==0, mask2==0).nonzero()]=np.nan
		else:	
			phi[(mask1==0).nonzero()]=np.nan

	condPara=np.logical_and(np.isfinite(phi), (180./3.14)*phi < 20.).nonzero() 
	nPara=np.size(phi[condPara])
	nGood=np.size(phi[np.isfinite(phi).nonzero()])

	corrframe=np.zeros(sz1)
	corrframe[condPara]=1.

	hogcorr=nPara/float(nGood)

        return hogcorr, corrframe

def HOGcorr_cube(cube1, cube2, z1min, z1max, z2min, z2max, ksz=1, mask1=0, mask2=0):

	sz1=np.shape(cube1)
	sz2=np.shape(cube2)

	corrplane=np.zeros([z1max-z1min, z2max-z2min])
	corrcube=np.zeros(sz1)

	for i in range(z1min, z1max):
                for j in range(z2min, z2max):
				
				if np.array_equal(np.shape(cube1), np.shape(mask1)):
					if np.array_equal(np.shape(cube2), np.shape(mask2)):				
						corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz, mask1=mask1[i,:,:], mask2=mask2[i,:,:])
					else:
						corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz, mask1=mask1[i,:,:])
				else:
					corr, corrframe=HOGcorr_frame(cube1[i,:,:], cube2[j,:,:], ksz=ksz)
				corrplane[i-z1min,j-z2min]=corr

	return corrplane


