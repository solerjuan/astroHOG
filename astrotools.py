#
#
#

from scipy import interpolate
import numpy as np

def SpiralArms(pitch_vec=np.array([12.1, 1.0, 8.7, 9.4]), beta0_vec=np.array([23., 24., 40., 18.]), R0_vec=np.array([4.91, 6.04, 8.87, 12.24]), beta1_vec=np.array([104., 97., 115., 71.]), num=100):

   num=100
   arms_r=np.zeros([np.size(pitch_vec),num])
   arms_phi=np.zeros([np.size(pitch_vec),num])

   for i in range(0,np.size(pitch_vec)):

      pitch_angle=pitch_vec[i]
      beta0=np.deg2rad(beta0_vec[i]) #*!pi/180.
      beta1=np.deg2rad(beta1_vec[i]) #*!pi/180.
      R0=R0_vec[i]    #         ; kpc
      b_coeff=np.tan(np.deg2rad(pitch_angle))
      a_coeff=R0/(np.exp(b_coeff*beta0))
      beta_vec=np.linspace(beta0, beta1, num=num)

      arms_phi[i,:]=beta_vec
      arms_r[i,:]  =a_coeff*np.exp(b_coeff*beta_vec)

   return arms_phi, arms_r



