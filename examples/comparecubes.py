# Example of the comparison of two data cubes using the astroHOG 
# 
# Prepared by Juan D. Soler (juandiegosoler@gmail.com)

import sys
sys.path.append('../')
from astrohog2d import *
from astrohog2d1v import *

import matplotlib.pyplot as plt
from astropy.io import fits

from scipy import ndimage

# Load the cubes that you plan to compare
# Just for reference, I assume that the first index runs over the non-spatial coordinate
# Also assume that the cubes are spatially aligned and are reprojected into the same grid 

hdul = fits.open('../data/testcube1.fits')
cube1=hdul[0].data
hdul.close()
hdul = fits.open('../data/testcube2.fits')
cube2=hdul[0].data
hdul.close()

fig, ax = plt.subplots(1,2, figsize=(16., 5.))
plt.rc('font', size=12)
ax[0].imshow(cube1.sum(axis=0), origin='lower', cmap='magma', interpolation='none')
ax[0].set_title('Integrated cube 1')
ax[0].tick_params(axis='y',labelrotation=90)
ax[1].imshow(cube2.sum(axis=0), origin='lower', cmap='viridis', interpolation='none')
ax[1].set_title('Integrated cube 2')
ax[1].tick_params(axis='y',labelrotation=90)
plt.show()

# Here you select the size of your derivative kernel in pixels
ksz=10

# Here I define the masks for both cubes
# For the sake of simplicity, I'm only masking the edges of the cubes
sz1=np.shape(cube1)
mask1=1.+0.*cube1
mask1[:,0:ksz,:]=0.
mask1[:,sz1[1]-1-ksz:sz1[1]-1,:]=0.
mask1[:,:,0:ksz]=0.
mask1[:,:,sz1[2]-1-ksz:sz1[2]-1]=0.
sz2=np.shape(cube2)
mask2=1.+0.*cube2
mask2[:,0:ksz,:]=0.
mask2[:,sz2[1]-1-ksz:sz2[1]-1,:]=0.
mask2[:,:,0:ksz]=0.
mask2[:,:,sz2[2]-1-ksz:sz2[2]-1]=0.

# Here you define the channel ranges over which you want to compare the cubes
zmin1=0
zmax1=sz1[0]-1
zmin2=0
zmax2=sz2[0]-1

# Statistical weights to account for the number of independent gradient pairs within a kernel
weights=(1./ksz)**2

# Run the HOG
circstats, corrcube, scube1, scube2 = HOGcorr_ppvcubes(cube1, cube2, zmin1, zmax1, zmin2, zmax2, ksz=ksz, mask1=mask1, mask2=mask2, weights=weights)

# The outputs are: 
# 1. 'corrplane' an array with all of the metrics to evaluate the correlation between the cubes
np.shape(circstats['V'])

# 2. 'corrcube', which is the array containing all the relative orientation angles between gradients
np.shape(corrcube)

# Plot the pair of channels with the highest spatial correlations
#
#
vplane=circstats['V']
vvec=np.sort(np.ravel(vplane))[::-1]
indmax1, indmax2 =(vplane == vvec[0]).nonzero()

# Here for example, we show the projected Rayleight statistic (V)
# Large V values indicate that the angle distribution is not flat and is centred on zero
# V values around zero correspond to a flat angle distribution.

fig, ax = plt.subplots(1,1, figsize=(6., 6.))
plt.rc('font', size=12)
im=ax.imshow(circstats['V'], origin='lower',clim=[0.,np.nanmax(circstats['V'])], interpolation='None')
ax.scatter(indmax1, indmax2, marker='*', color='magenta')
ax.set_xlabel('Channel in cube 2')
ax.set_ylabel('Channel in cube 1')
ax.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im, fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
plt.show()

fig, ax = plt.subplots(1,2, figsize=(16., 5.))
ax[0].imshow(scube1[indmax1[0],:,:], origin='lower', cmap='magma', interpolation='none')
ax[0].set_title('Image 1')
ax[0].tick_params(axis='y',labelrotation=90)
ax[1].imshow(scube2[indmax2[0],:,:], origin='lower', cmap='viridis', interpolation='none')
ax[1].set_title('Image 2')
ax[1].tick_params(axis='y',labelrotation=90)
plt.show()

# Plot the relative orientation angle between the gradients in the two images with the highest correlation
#
#
fig, ax = plt.subplots(1,1, figsize=(8., 5.))
im=ax.imshow(np.abs(corrcube[indmax1[0],indmax2[0],:,:])*180.0/np.pi, origin='lower', cmap='spring',interpolation='None')
ax.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im,fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$|\phi|$ [deg]')
cbl.ax.tick_params(axis='y',labelrotation=90)
ax.set_title('Relative orientation between gradient vectors')
plt.show()

# Making block average with block size matching the kernel size
sz=np.shape(cube1)
nbyA=8; nbxA=int(np.ceil(nbyA*sz[1]/sz[2]))
vblocksA=imablockaverage(corrcube[indmax1[0],indmax2[0],:,:], nbx=nbxA, nby=nbyA, weight=(1./ksz)**2)

# Making block average with block size matching the kernel size
nbyB=int(0.25*sz[2]/ksz); nbxB=int(nbyB*sz[1]/sz[2])
vblocksB=imablockaverage(corrcube[indmax1[0],indmax2[0],:,:], nbx=nbxB, nby=nbyB, weight=(1./ksz)**2)

fig = plt.figure(figsize=(6.0, 8.0))
plt.rc('font', size=10)
ax1=plt.subplot(211)
ax1.set_title(str(nbxA)+'x'+str(nbyA)+' blocks')
im=ax1.imshow(vblocksA, origin='lower', vmin=0., interpolation='none', aspect='auto')
ax1.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im,fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
ax2=plt.subplot(212)
ax2.set_title(str(nbxB)+'x'+str(nbyB)+' blocks')
im=ax2.imshow(vblocksB, origin='lower', vmin=0., interpolation='none', aspect='auto')
ax2.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im,fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
plt.show()

# Plot the pair of channels with the second highest spatial correlations
#
#
indmax1, indmax2 =(vplane == vvec[1]).nonzero()

fig, ax = plt.subplots(1,2, figsize=(15., 5.))
ax[0].imshow(scube1[indmax1[0],:,:], origin='lower', cmap='magma', interpolation='none')
ax[0].set_title('Image 1')
ax[0].tick_params(axis='y',labelrotation=90)
ax[1].imshow(scube2[indmax2[0],:,:], origin='lower', cmap='viridis', interpolation='none')
ax[1].set_title('Image 2')
ax[1].tick_params(axis='y',labelrotation=90)
plt.show()

# Running the jackknife tests
circstats01, corrcube01, scube1j01, scube2j01 = HOGcorr_ppvcubes(cube1, cube2[:,:,::-1], zmin1, zmax1, zmin2, zmax2, ksz=ksz, mask1=mask1, mask2=mask2, weights=weights)
vplane01=circstats01['V']
circstats10, corrcube10, scube1j10, scube2j10 = HOGcorr_ppvcubes(cube1, cube2[:,::-1,:], zmin1, zmax1, zmin2, zmax2, ksz=ksz, mask1=mask1, mask2=mask2, weights=weights)
vplane10=circstats10['V']
circstats11, corrcube11, scube1j11, scube2j11 = HOGcorr_ppvcubes(cube1, cube2[:,::-1,::-1], zmin1, zmax1, zmin2, zmax2, ksz=ksz, mask1=mask1, mask2=mask2, weights=weights)
vplane11=circstats11['V']

# Getting maximum value of the projected Rayleigh statistic (V) for the plots
maxV=np.nanmax([vplane,vplane01,vplane10,vplane11])
vvec=np.sort(np.ravel(vplane))[::-1]
indmax1, indmax2 =(vplane == vvec[0]).nonzero()

# Plotting jackknife tests
#
#
fig = plt.figure(figsize=(10.0, 10.0))
plt.rc('font', size=10)
ax1=plt.subplot(221)
ax1.set_title('Original data')
im=ax1.imshow(vplane, origin='lower',clim=[0.,maxV], interpolation='None')
ax1.scatter(indmax1, indmax2, marker='*', color='magenta')
ax1.set_xlabel(' '); ax1.set_ylabel('Channel in cube 1')
ax1.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im, fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
ax2=plt.subplot(222)
ax2.set_title('Vertical flipping')
im=ax2.imshow(vplane10, origin='lower',clim=[0.,maxV], interpolation='None')
ax2.scatter(indmax1, indmax2, marker='*', color='magenta')
ax2.set_xlabel(' '); ax2.set_ylabel(' ')
ax2.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im, fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
ax3=plt.subplot(223)
ax3.set_title('Horizontal flipping')
im=ax3.imshow(vplane01, origin='lower',clim=[0.,maxV], interpolation='None')
ax3.scatter(indmax1, indmax2, marker='*', color='magenta')
ax3.set_xlabel('Channel in cube 2'); ax3.set_ylabel('Channel in cube 1')
ax3.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im, fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
ax4=plt.subplot(224)
ax4.set_title('Horizontal and vertical flipping')
im=ax4.imshow(vplane11, origin='lower',clim=[0.,maxV], interpolation='None')
ax4.scatter(indmax1, indmax2, marker='*', color='magenta')
ax4.set_xlabel('Channel in cube 2'); ax4.set_ylabel(' ')
ax4.tick_params(axis='y',labelrotation=90)
cbl=plt.colorbar(im, fraction=0.046, pad=0.04)
cbl.ax.set_title(r'$V$')
cbl.ax.tick_params(axis='y',labelrotation=90)
plt.show()



