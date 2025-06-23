import numpy as np
import matplotlib.pyplot as plt
from Utils.styles import *
import pickle
from Greitzer.src.config import Config
from scipy.optimize import fsolve
import os

PICS_FOLDER = 'pics'
RESULT_FOLDER = 'results'


class Greitzer:
    """
    Class that creates a greitzer model
    """

    def __init__(self, config):
        """
        :param systemData: parameters of the system
        """
        self.config = config
        


    def computeCompressorValveIntersection(self):
        
        phi = self.config.get_flow_coeffs()
        psi_c = self.config.get_work_coeffs()
        k_valve = self.config.get_valve_coefficient()
        
        #polynomial interpolation for the compressor curve
        z_coeff = np.polyfit(phi, psi_c, 3)
        phi = np.linspace(0, 1, 1000)
        psi_c = np.polyval(z_coeff, phi)

        #throttle valve curve
        psi_v = k_valve*phi**2
        
        def residualFunction(phi):
            return np.polyval(z_coeff, phi) - k_valve*phi**2
        
        initial_phi_guess = 0.9
        phi_eq = fsolve(residualFunction, initial_phi_guess)
        psi_eq = k_valve * phi_eq ** 2
        
        #calculate derivatives of the compressor and throttle characteristic at equilibrium point
        phi_eq = phi_eq[0]       # unpack the scalar value from the array
        psi_eq = psi_eq[0]
        delta_phi = phi_eq * 0.001

        # Recompute based on unpacked scalar phi_eq
        phi_left = phi_eq - delta_phi
        phi_right = phi_eq + delta_phi
        psi_c_right = np.polyval(z_coeff, phi_right)
        psi_c_left = np.polyval(z_coeff, phi_left)
        psi_c_prime = (psi_c_right - psi_c_left) / (2 * delta_phi)
        psi_v_prime = 2 * k_valve * phi_eq
        
        return phi_eq, psi_eq, psi_c_prime, psi_v_prime
    
    
    
    def computeLinearizedStabilityMap(self):
        resolution = 50
        B_min = 0.001
        B_max = 1.25
        G_min = 0.001
        G_max = 3
        B = np.linspace(B_min,B_max,resolution)
        G = np.linspace(G_min,G_max,resolution)
        self.B_grid, self.G_grid = np.meshgrid(B, G, indexing='ij')
        self.stabilityMap = np.zeros((resolution,resolution))
        
        phi_eq, psi_eq, psi_c_prime, psi_v_prime = self.computeCompressorValveIntersection()
        self.B_system, self.G_system = self.compute_B_G_params()

        for i in range(len(B)):
            for j in range(len(G)):
                B_r = self.B_grid[i,j]
                G_r = self.G_grid[i,j]
                coeffs = [-1, 
                        B_r*psi_c_prime-B_r*psi_v_prime/G_r,
                        (psi_c_prime*psi_v_prime*B_r**2)/G_r - 1/G_r -1, 
                        (B_r/G_r)*(psi_c_prime-psi_v_prime)]
                
                # Find the roots of the polynomial
                roots = np.roots(coeffs)
                roots_real = roots.real

                if (roots_real[0]>=0 or roots_real[1]>=0 or roots_real[2]>=0):
                    self.stabilityMap[i,j] = 1 #unstable
                else:
                    self.stabilityMap[i,j] = 0 #stable
    
    
    
    def plotStabilityMap(self, save_filename=None):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        plt.figure()
        plt.contourf(self.B_grid, self.G_grid, self.stabilityMap, cmap='bwr')
        plt.plot(self.B_system, self.G_system, 'ow')
        plt.xlabel(r'$B$')
        plt.ylabel(r'$G$')
        if save_filename is not None:
            plt.savefig(PICS_FOLDER + '/' + save_filename + '_stability_map_B_%.3f.pdf' % (self.B_system), bbox_inches='tight')
        
    
        
    def compute_B_G_params(self):
        U_ref = self.config.get_reference_speed()
        a = self.config.get_sound_speed()
        Vp = self.config.get_plenum_volume()
        Lc = self.config.get_inlet_duct_length()
        Lt = self.config.get_throttle_duct_length()
        At = self.config.get_throttle_duct_area()
        Ac = self.config.get_inlet_duct_area()
        
        B_real = 0.3*(U_ref/(2*a))*np.sqrt(Vp/(Ac*Lc))  
        G_real = Lt*Ac/(Lc*At)                      
        
        return B_real, G_real
    
    def savePickle(self, save_filename):
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        
        with open(RESULT_FOLDER + '/' + save_filename + '.pkl', 'wb') as handle:
            pickle.dump(self, handle)
        
        print()
        print('Results saved in file: ' + RESULT_FOLDER + '/' + save_filename + '.pkl')
        print()