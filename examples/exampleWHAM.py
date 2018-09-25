# This file is part of AstroHOG
#
# Copyright (C) 2013-2017 Juan Diego Soler

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

sys.path.append('/Users/jsoler/Documents/astrohog/')
from astrohog import *

from astropy.wcs import WCS
from reproject import reproject_interp

def astroHOGexampleWHAM(frame, vmin, vmax, ksz=1):
	fstr="%4.2f" % frame

	dir='/Users/jsoler/DATA/WHAM/'
	hdu1=fits.open(dir+'hi_filament_cube.fits')
	hdu2=fits.open(dir+'ha_filament_cube.fits')

	v1=vmin*1000.;	v2=vmax*1000.
        v1str="%4.1f" % vmin     
	v2str="%4.1f" % vmax
	limsv=np.array([v1, v2, v1, v2])

	cube1=hdu1[0].data
	sz1=np.shape(hdu1[0].data)
	CTYPE3=hdu1[0].header['CTYPE3']
	CDELT3=hdu1[0].header['CDELT3']
	CRVAL3=hdu1[0].header['CRVAL3']
	CRPIX3=hdu1[0].header['CRPIX3']
	#zmin1=0        
	zmin1=int(CRPIX3+(v1-CRVAL3)/CDELT3)
	#zmax1=sz1[0]-1
	zmax1=int(CRPIX3+(v2-CRVAL3)/CDELT3)
	velvec1=hdu1[0].header['CRVAL3']+(np.arange(sz1[0])-hdu1[0].header['CRPIX3'])*hdu1[0].header['CDELT3']
        #np.arange(v1,v2,CDELT3)/1000.
	
	cube2=hdu2[0].data
	sz2=np.shape(hdu2[0].data)
        CTYPE3=hdu2[0].header['CTYPE3']
        CDELT3=hdu2[0].header['CDELT3']
        CRVAL3=hdu2[0].header['CRVAL3']
        CRPIX3=hdu2[0].header['CRPIX3']
        #zmin2=0
	zmin2=int(CRPIX3+(v1-CRVAL3)/CDELT3)
        #zmax2=sz2[0]-1  
	zmax2=int(CRPIX3+(v2-CRVAL3)/CDELT3)
        velvec2=hdu2[0].header['CRVAL3']+(np.arange(sz2[0])-hdu2[0].header['CRPIX3'])*hdu2[0].header['CDELT3'] 

	refhdr1=hdu1[0].header.copy()
        NAXIS31=refhdr1['NAXIS3']
        del refhdr1['NAXIS3']
        del refhdr1['CTYPE3']
        del refhdr1['CRVAL3']
        del refhdr1['CRPIX3']
        del refhdr1['CDELT3']
        del refhdr1['CUNIT3']
	del refhdr1['CNAME3']
        refhdr1['NAXIS']=2
	refhdr1['WCSAXES']=2

	refhdr2=hdu2[0].header.copy()
        NAXIS3=refhdr2['NAXIS3']
	del refhdr2['NAXIS3']
        del refhdr2['CTYPE3']
	del refhdr2['CRVAL3']
	del refhdr2['CRPIX3']
	del refhdr2['CDELT3']
	del refhdr2['CUNIT3']
	del refhdr2['CNAME3']
	del refhdr2['PV1_3']
        refhdr2['NAXIS']=2
	refhdr2['WCSAXES']=2

	newcube1=np.zeros([NAXIS31, sz2[1], sz2[2]])		

	for i in range(0, NAXIS31):
		hduX=fits.PrimaryHDU(cube1[i,:,:])
		hduX.header=refhdr1
		mapX, footprintX=reproject_interp(hduX, refhdr2)

		newcube1[i,:,:]=mapX

	#import pdb; pdb.set_trace()

	# ==========================================================================================================
	sz1=np.shape(newcube1)
	x=np.sort(newcube1.ravel())
  	minrm=x[int(0.2*np.size(x))]
	#minrm=np.std(newcube1[0,:,:])
	mask1=np.zeros(sz1)
	mask1[(newcube1 > minrm).nonzero()]=1
	mask1[:,0:ksz,:]=0.; mask1[:,sz1[1]-ksz:sz1[1],:]=0.
	mask1[:,:,0:ksz]=0.; mask1[:,:,sz1[2]-ksz:sz1[2]]=0.
	mask1[:,sz1[1]-80:sz1[1],:]=0.;

	sz2=np.shape(cube2)
	minrm=np.std(cube2[0,:,:])
	mask2=np.zeros(sz2)
	mask2[(cube2 > minrm).nonzero()]=1
	
	corrplane, corrcube=HOGcorr_cube(newcube1, cube2, zmin1, zmax1, zmin2, zmax2, ksz=ksz, mask1=mask1, mask2=mask2)
	
	strksz="%i" % ksz

	limsv=np.array([velvec1[zmin1], velvec1[zmax1], velvec2[zmin2], velvec2[zmax2]])
	plt.imshow(corrplane, origin='lower', extent=limsv/1e3, interpolation='none')
	plt.xlabel(r'$v_{HI}$ [km/s]')
        plt.ylabel(r'$v_{H\alpha}$ [km/s]')
        plt.yticks(rotation='vertical')
	plt.colorbar()
	plt.show()
	#plt.savefig('HOGcorrelationPlanck353GRSL'+fstr+'_b'+blimstr+'_k'+strksz+'_v'+v1str+'to'+v2str+'.png', bbox_inches='tight')
        #plt.close()

	ix=(corrplane == np.max(corrplane)).nonzero()[0][0]		
	jx=(corrplane == np.max(corrplane)).nonzero()[1][0] 
	print(velvec1[ix]/1e3)
	print(velvec2[jx]/1e3)

	#limsv=np.array([velvec1[ix-10], velvec1[ix+10], velvec2[jx-10], velvec2[jx+10]])
        #plt.imshow(corrplane[ix-10:ix+10,jx-10:jx+10], origin='lower', extent=limsv/1e3, interpolation='none')
        #plt.xlabel(r'$v_{HI}$ [km/s]')
        #plt.ylabel(r'$v_{H\alpha}$ [km/s]')
        #plt.yticks(rotation='vertical')
        #plt.colorbar()
        #plt.show()

	ax1=plt.subplot(1,1,1, projection=WCS(refhdr2))
	ax1.imshow(newcube1[ix,:,:], origin='lower', cmap='seismic', clim=[np.min(newcube1[ix,:,:]),4.]) #, interpolation='none')
	ax1.imshow(cube2[jx,:,:],    origin='lower', alpha=0.55, cmap='binary', clim=[0.,1.0])  
	ax1.coords.grid(color='white')
	ax1.coords['glon'].set_axislabel('Galactic Longitude')
	ax1.coords['glat'].set_axislabel('Galactic Latitude')
	ax1.coords['glat'].set_axislabel_position('r')
	ax1.coords['glat'].set_ticklabel_position('r')
	ax1.set_title('DKs cubes')
	plt.show()

	inmap=newcube1[ix,:,:]
	inmap[inmap > np.mean(inmap)]=np.mean(inmap)
        r=(inmap-np.min(inmap))/(np.max(inmap)-np.min(inmap))
	inmap=cube2[jx,:,:]
	inmap[inmap > np.mean(inmap)]=np.mean(inmap)
        g=(inmap-np.min(inmap))/(np.max(inmap)-np.min(inmap))

	b=0.*g
		
	sz=np.shape(r)
        rgb=np.zeros([sz[0], sz[1], 3])
        rgb[:,:,0]=r#(1.-r)
        rgb[:,:,1]=g#(1.-g)
        rgb[:,:,2]=b

	ax1=plt.subplot(1,1,1, projection=WCS(refhdr2))
	ax1.imshow(rgb, origin='lower')
	ax1.coords['glon'].set_axislabel('Galactic Longitude')
        ax1.coords['glat'].set_axislabel('Galactic Latitude')
        ax1.coords['glat'].set_axislabel_position('r')
        ax1.coords['glat'].set_ticklabel_position('r')
        ax1.set_title('DKs cubes')
        plt.show()

	ax1=plt.subplot(1,1,1, projection=WCS(refhdr2))
        ax1.imshow(corrcube[ix,:,:], origin='lower', cmap='Reds', clim=[np.min(newcube1[ix,:,:]),4.])
        plt.show()

	corrcube[np.isnan(corrcube).nonzero()]=0.
	ax1=plt.subplot(1,1,1, projection=WCS(refhdr2))
	ax1.imshow(corrcube[ix-1:ix+1,:,:].sum(axis=0), origin='lower', cmap='seismic')
	plt.show()
        import pdb; pdb.set_trace()


ksz=5
astroHOGexampleWHAM(23.75, 0., 45., ksz=ksz)
#astroHOGexampleWHAM(23.75, -45., 45., ksz=ksz)




