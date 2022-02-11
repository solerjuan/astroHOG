"""
astroHOG Statistical tests
"""

import numpy as np

# ------------------------------------------------------------------------------------------------------------------------
def HOG_PRS(phi, weights=None):
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

   if weights is None:
      weights=np.ones_like(phi)

   angles=phi #2.*phi

   circX=np.sum(weights*np.cos(angles))/np.sum(weights)
   circY=np.sum(weights*np.sin(angles))/np.sum(weights)
   mrv=np.sqrt(circX**2+circY**2)

   Zx=np.sum(weights*np.cos(angles))/np.sqrt(np.sum(weights)/2.)
   #Zx=np.sum(np.cos(angles))/np.sqrt(np.size(angles)/2.)
   temp=np.sum(np.cos(angles)*np.cos(angles))
   s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))

   Zy=np.sum(weights*np.sin(angles))/np.sqrt(np.sum(weights)/2.)
   #Zy=np.sum(np.sin(angles))/np.sqrt(np.size(angles)/2.)
   temp=np.sum(np.sin(angles)*np.sin(angles))
   s_Zy=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))

   meanPhi=0.5*np.arctan2(Zy, Zx)

   #return Zx, s_Zx, meanPhi
   return {'Zx': Zx, 's_Zx': s_Zx, 'meanphi': meanPhi, 'mrv': mrv}

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


