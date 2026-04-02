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
              'meanphi': output['meanphi'], 'stdphi': output['stdphi'], 's_meanphi': np.nan, 
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
      stdphi=np.mean(arrstdphi)
      s_meanphi=circstd(arrmeanphi, low=-np.pi, high=np.pi)  
      mrv  =np.nanmean(arrmrv)
      s_mrv=np.nanstd(arrmrv)
     
      return {'Z': Z, 's_Z': s_Z, 'Zx': Zx, 's_Zx': s_Zx, 's_ZxMC': s_ZxMC, 'meanphi': meanphi, 'stdphi': stdphi, 's_meanphi': s_meanphi, 'mrv': mrv, 's_mrv': s_mrv, 'ngood': ngood} 

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

   if (np.size(angles) < 1):
      return {'Z': np.nan, 's_Z': np.nan, 'Zx': np.nan, 's_Zx': np.nan, 'meanphi': np.nan, 'stdphi': np.nan, 'mrv': np.nan, 'ngood': np.size(angles)}

   if weights is None:
      weivec=np.ones_like(angles)
   elif(np.size(weights)==1):
      weivec=weights*np.ones_like(angles)
   elif(np.size(angles)==np.size(weights)):
      weivec=weights.copy()
   else:
      diff=np.abs(np.size(angles)-np.size(weights))   
      if (diff > 0.):
         print("HOGPRS: Weights vectos should have the same dimensions as angle vector")
         return 0
         
   circX=np.sum(weivec*np.cos(angles))/np.sum(weivec)
   circY=np.sum(weivec*np.sin(angles))/np.sum(weivec)
   mrv=np.sqrt(circX**2+circY**2)

   Zx=np.sum(weivec*np.cos(angles))/np.sqrt(np.sum(weivec)/2.)
   temp=np.sum(np.cos(angles)*np.cos(angles))
   if ((2.*temp-Zx*Zx) > 0.):
      s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))
   else: 
      s_Zx=np.nan

   Zy=np.sum(weivec*np.sin(angles))/np.sqrt(np.sum(weivec)/2.)
   temp=np.sum(np.sin(angles)*np.sin(angles))
   if ((2.*temp-Zy*Zy) > 0.):
      s_Zy=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))
   else: 
      s_Zy=np.nan

   Z=np.sqrt(Zx**2+Zy**2)
   s_Z=np.sqrt(s_Zx**2+s_Zy**2)

   meanphi=circmean(angles, low=-np.pi, high=np.pi)
   #meanphi=np.arctan(Zy/Zx)
   stdphi=circstd(angles, low=-np.pi, high=np.pi)
   varphi=1-mrv
   #stdphi=np.sqrt(np.log(1/mrv**2))

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

# ---------------------------------------------------------------------
def kuiper_two_sample(x, y):
    """
    Two-sample Kuiper test for circular data.
    x, y: angles in radians on [0, 2pi).
    Returns (V, p-value).
    """
    x = np.mod(x, 2*np.pi)
    y = np.mod(y, 2*np.pi)

    n = len(x)
    m = len(y)

    data = np.concatenate([x, y])
    labels = np.concatenate([np.zeros(n), np.ones(m)])

    order = np.argsort(data)
    labels = labels[order]

    # Empirical CDFs
    cdf_x = np.cumsum(labels == 0) / n
    cdf_y = np.cumsum(labels == 1) / m

    Dplus =  np.max(cdf_x - cdf_y)
    Dminus = np.max(cdf_y - cdf_x)
    V = Dplus + Dminus

    # Stephens approximation for p-value
    eff_n = n * m / (n + m)
    lam = (np.sqrt(eff_n) + 0.155 + 0.24/np.sqrt(eff_n)) * V

    def Q(l):
        s = 0.0
        for j in range(1, 101):
            s += (4*j*j*l*l - 1) * np.exp(-2*j*j*l*l)
        return 2*s

    pval = Q(lam)
    return V, pval

# -----------------------------------------------------------------
def axial_to_directional(angles_deg):
    """Convert orientation angles (deg, -90..90) to directional angles (rad, 0..2π)."""

    theta = np.deg2rad(angles_deg)
    return (2 * theta) % (2 * np.pi)

# -----------------------------------------------------------------
def map_angle_pi_to_halfpi(angle):
    """
    Maps angles from [-π, π] into [-π/2, π/2] by wrapping.
    Accepts scalars or NumPy arrays.
    """
    angle = np.asarray(angle)

    # Normalize to [-π, π]
    angle_norm = (angle + np.pi) % (2 * np.pi)
    angle_norm = np.where(angle_norm < 0, angle_norm + 2 * np.pi, angle_norm)
    angle_norm = angle_norm - np.pi

    half_pi = np.pi / 2

    # Wrap values outside [-π/2, π/2]
    wrapped = np.where(angle_norm > half_pi,
                       angle_norm - np.pi,
                       np.where(angle_norm < -half_pi,
                                angle_norm + np.pi,
                                angle_norm))
    return wrapped

# -----------------------------------------------------------------
def rotate_and_wrap_90(angle):
    """
    Rotates an angle (in radians) by +90 degrees (π/2)
    and wraps it into the range [-π/2, π/2].
    Works for scalars or NumPy arrays.
    """
    angle = np.asarray(angle)

    # Rotate by +90 degrees (π/2 radians)
    rotated = angle + np.pi / 2

    # Normalize to [-π, π]
    norm = (rotated + np.pi) % (2 * np.pi)
    norm = np.where(norm < 0, norm + 2 * np.pi, norm)
    norm = norm - np.pi

    # Wrap into [-π/2, π/2]
    half_pi = np.pi / 2
    wrapped = np.where(norm > half_pi,
                       norm - np.pi,
                       np.where(norm < -half_pi,
                                norm + np.pi,
                                norm))
    return wrapped




