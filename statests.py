"""
astroHOG Statistical tests
"""

import numpy as np

# ------------------------------------------------------------------------------------------------------------------------
def HOG_PRS(phi):
   # Calculates the projected Rayleigh statistic of the distributions of angles phi.
   #
   # INPUTS
   # phi      - angles between -pi/2 and pi/2
   #
   # OUTPUTS
   # Zx       - value of the projected Rayleigh statistic   
   # s_Zx     - 
   # meanPhi  -

   angles=phi #2.*phi

   Zx=np.sum(np.cos(angles))/np.sqrt(np.size(angles)/2.)
   temp=np.sum(np.cos(angles)*np.cos(angles))
   s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))

   Zy=np.sum(np.sin(angles))/np.sqrt(np.size(angles)/2.)
   temp=np.sum(np.sin(angles)*np.sin(angles))
   s_Zx=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))

   meanPhi=0.5*np.arctan2(Zy, Zx)

   return Zx, s_Zx, meanPhi

# ------------------------------------------------------------------------------------------------------------------------------
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


