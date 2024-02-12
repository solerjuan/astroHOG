# #!/usr/bin/env python
#
# This file is part of astroHOG
#
# CONTACT: juandiegosolerp[at]gmail.com
# Copyright (C) 2017-2023 Juan Diego Soler

"""
astroHOG Statistical tests
"""

import numpy as np
from scipy.stats import circmean, circstd
import pycircstat

# -------------------------------------------------------------------------------------------------------
def HOG_PRS(phi, weights=None, s_phi=None, nruns=1):
   # Calculates the projected Rayleigh statistic of the distributions of angles phi.
   #
   # INPUTS
   # phi      - angles between -pi/2 and pi/2
   # weights  - statistical weights
   #
   # OUTPUTS
   # Zx       - value of the projected Rayleigh statistic   
   # s_Zx     - 
   # meanPhi  -

   if np.logical_or(s_phi is None, nruns<2):

      output=HOG_PRSlite(phi, weights=weights)
      return {'Z': output['Z'], 's_Z': output['s_Z'], 
              'Zx': output['Zx'], 's_Zx': output['s_Zx'], 's_ZxMC': np.nan, 
              'meanphi': output['meanphi'], 's_meanphi': np.nan, 
              'mrv': output['mrv'], 's_mrv': np.nan, 'ngood': output['ngood']}

   else: 

      arrngood=np.zeros(nruns)
      arrZ=np.zeros(nruns)
      arrs_Z=np.zeros(nruns)
      arrZx=np.zeros(nruns)
      arrs_Zx=np.zeros(nruns)   
      arrmeanphi=np.zeros(nruns)
      arrstdphi=np.zeros(nruns)
      arrmrv=np.zeros(nruns)
 
      for i in range(0,nruns):

         inmaprand=np.random.normal(loc=phi, scale=s_phi)  
         output=HOG_PRSlite(inmaprand, weights=weights)
         arrngood[i]=output['ngood']
         arrZ[i]=output['Z']
         arrs_Z[i]=output['s_Z']
         arrZx[i]=output['Zx']
         arrs_Zx[i]=output['s_Zx']
         arrmeanphi[i]=output['meanphi']
         arrstdphi[i]=output['stdphi']
         arrmrv[i]=output['mrv']
    
      ngood=np.nanmean(arrngood)
      Z     =np.nanmean(arrZ)
      s_Z   =np.nanmean(arrs_Z) 
      Zx    =np.nanmean(arrZx)
      s_Zx  =np.nanmean(arrs_Zx)
      s_ZxMC=np.nanstd(arrZx) 
      meanphi =circmean(arrmeanphi, low=-np.pi, high=np.pi)
      s_meanphi=circstd(arrmeanphi, low=-np.pi, high=np.pi)  
      mrv  =np.nanmean(arrmrv)
      s_mrv=np.nanstd(arrmrv)
     
      return {'Z': Z, 's_Z': s_Z, 'Zx': Zx, 's_Zx': s_Zx, 's_ZxMC': s_ZxMC, 'meanphi': meanphi, 's_meanphi': s_meanphi, 'mrv': mrv, 's_mrv': s_mrv, 'ngood': ngood} 

# ------------------------------------------------------------------------------------------------------------------------
def HOG_PRSlite(angles, weights=None):
   # Calculates the projected Rayleigh statistic of the distributions of angles phi.
   #
   # INPUTS
   # angles   - angles between -pi/2 and pi/2
   # weights  - statistical weights
   #
   # OUTPUTS
   # Zx       - value of the projected Rayleigh statistic   
   # s_Zx     - 
   # meanPhi  -

   if weights is None:
      weights=np.ones_like(angles)

   #angles=phi #2.*phi

   circX=np.sum(weights*np.cos(angles))/np.sum(weights)
   circY=np.sum(weights*np.sin(angles))/np.sum(weights)
   mrv=np.sqrt(circX**2+circY**2)

   #p0, Zx0=pycircstat.tests.vtest(angles, 0., w=weights)
   #print("Zx0", Zx0/np.sqrt(np.sum(weights)/2.)) # Too match the Jow et al. (2018) results
   Zx=np.sum(weights*np.cos(angles))/np.sqrt(np.sum(weights**2)/2.)
   temp=np.sum(np.cos(angles)*np.cos(angles))
   s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))

   Zy=np.sum(weights*np.sin(angles))/np.sqrt(np.sum(weights**2)/2.)
   temp=np.sum(np.sin(angles)*np.sin(angles))
   s_Zy=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))

   Z=np.sqrt(Zx**2+Zy**2)
   s_Z=np.sqrt(s_Zx**2+s_Zy**2)

   meanphi=circmean(angles, low=-np.pi, high=np.pi)
   stdphi=circstd(angles, low=-np.pi, high=np.pi)

   ngood=float(np.size(angles)) 

   #import pdb; pdb.set_trace()
   #return Zx, s_Zx, meanPhi
   return {'Z': Z, 's_Z': s_Z, 'Zx': Zx, 's_Zx': s_Zx, 'meanphi': meanphi, 'stdphi': stdphi, 'mrv': mrv, 'ngood': ngood}

# ---------------------------------------------------------------------------------------------------------
def HOG_AM(phi):
   # Calculate the alignment measure.
   #
   # INPUTS
   # phi      - angles between -pi/2 and pi/2
   #
   # OUTPUTS
   #AM        - value of the alignment measure.  
 
   angles=phi

   ami=2.*np.cos(phi)-1.
   am=np.mean(ami)

   return am

# ---------------------------------------------------------------------------------------------------------
def CrossCorrelation(map1, map2, mask1=None, mask2=None):

   # Calculate cross correlation
   #
   # INPUTS
   # map1 
   # map2
   #
   # OUTPUTS
   # 

   if (mask1 is None):
      mask1=np.ones_like(map1)
   if (mask2 is None):
      mask2=np.ones_like(map2)

   bad1=np.isnan(map1).nonzero()
   mask1[bad1]=0.
   bad2=np.isnan(map2).nonzero()
   mask2[bad2]=0.

   good=np.logical_and(mask1 > 0., mask2 > 0.).nonzero()

   prod12=map1*map2
   rho12=np.sum(prod12[good])/np.sqrt(np.sum(map1[good]**2)*np.sum(map2[good]**2))   

   return rho12

# ---------------------------------------------------------------------------------------------------------
def PearsonCorrelationCoefficient(map1, map2, mask1=None, mask2=None):

   # Calculate cross correlation
   #
   # INPUTS
   # map1 
   # map2
   #
   # OUTPUTS
   # 

   if (mask1 is None):
      mask1=np.ones_like(map1)
   if (mask2 is None):
      mask2=np.ones_like(map2)

   bad1=np.isnan(map1).nonzero()
   mask1[bad1]=0.
   bad2=np.isnan(map2).nonzero()
   mask2[bad2]=0.

   good=np.logical_and(mask1 > 0., mask2 > 0.).nonzero()

   mean1=np.mean(map1[good])
   mean2=np.mean(map2[good])

   prod12=(map1-mean1)*(map2-mean2)
   rho12=np.sum(prod12[good])/np.sqrt(np.sum((map1[good]-mean1)**2)*np.sum((map2[good]-mean2)**2))

   return rho12


