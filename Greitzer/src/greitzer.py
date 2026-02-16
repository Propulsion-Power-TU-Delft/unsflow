import numpy as np
import matplotlib.pyplot as plt
from Utils.styles import *
import pickle
from Greitzer.src.config import Config
from scipy.optimize import fsolve
import os
from scipy.integrate import odeint

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
        resolution = 100
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

                self.stabilityMap[i,j] = np.max(roots_real)
                # if (roots_real[0]>=0 or roots_real[1]>=0 or roots_real[2]>=0):
                #     self.stabilityMap[i,j] = 1 #unstable
                # else:
                #     self.stabilityMap[i,j] = 0 #stable
    
    
    
    def plotStabilityMap(self, save_filename=None, plotSystem=False):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        plt.figure()
        css = plt.contourf(self.B_grid, self.G_grid, self.stabilityMap, cmap='RdBu_r', levels=15)
        cbar = plt.colorbar(css)
        cbar.set_label(r'max Re($\lambda$)')
        cs = plt.contour(self.B_grid, self.G_grid, self.stabilityMap, levels=[0], colors='k', linestyles='--', linewidths=2.0)
        plt.clabel(cs, fmt='%1.1f', inline=True, fontsize=18)
        if plotSystem:
            plt.plot(self.B_system, self.G_system, 'ow')
        plt.xlabel(r'$B$')
        plt.ylabel(r'$G$')
        # plt.title(r'max Re($\lambda$)')
        plt.tight_layout()
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
        
        B_real = (U_ref/(2*a))*np.sqrt(Vp/(Ac*Lc))  
        G_real = Lt*Ac/(Lc*At)                      
        
        return B_real, G_real
    
    def savePickle(self, save_filename):
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        
        with open(RESULT_FOLDER + '/' + save_filename + '.pkl', 'wb') as handle:
            pickle.dump(self, handle)
        
        print()
        print('Results saved in file: ' + RESULT_FOLDER + '/' + save_filename + '.pkl')
        print()
    
    
    def solveGreitzerSystem(self):
        self.H_param, self.W_param, self.psi_c_0_param = self.config.get_unstalled_characteristic_params()
        self.phi = np.linspace(0,1,1000)
        self.psi_c = self.unstalled_characteristic(self.phi, self.H_param, self.W_param, self.psi_c_0_param)
        
        k_valve = self.config.get_valve_coefficient()
        self.psi_v = k_valve*self.phi**2
        
        def func_work_coefficient(phi):
            """
            Function needed by fsolve in order to find the intersection between unstalled
            compressor curve and throttle line
            """
            return self.unstalled_characteristic(phi, self.H_param, self.W_param, self.psi_c_0_param) - k_valve*phi**2

        initial_phi_guess = self.phi.min() + (self.phi.max()-self.phi.min())*0.5 
        phi_eq = fsolve(func_work_coefficient,initial_phi_guess) 
        psi_eq = k_valve*phi_eq**2 
        
        #time span
        t = np.linspace(0,self.config.get_max_time(),2500)
        a = self.config.get_sound_speed()
        Ac = self.config.get_inlet_duct_area()
        Vp = self.config.get_plenum_volume()
        Lc = self.config.get_inlet_duct_length()
        self.xi = t*a*np.sqrt(Ac/(Vp*Lc))
        
        #initial conditions
        perturbation = 1e-2
        y0 = [phi_eq[0]*(1-perturbation), 
            phi_eq[0]*(1+perturbation), 
            psi_eq[0]*(1-0.5*perturbation)] 
        
        self.B_system, self.G_system = self.compute_B_G_params()
        
        self.solutionGreitzer = odeint(self.greitzer3DSystem, y0, self.xi, args=(self.B_system, self.G_system, k_valve))
        

    def greitzer3DSystem(self, y, xi, B, G, k_valve):
        """
        Defines the differential equations for the Greitzer model
        (as found in the thesis of Sündstrom).

        Arguments:
            y :  vector of the state variables
            xi : non dimensional time
            B :  B parameter
            G :  G parameter
            
        State variables:
            x1 : compressor flow coefficient
            x2 : throttle flow coefficient
            x3 : compressor work coefficient
        """
        x1, x2, x3 = y
        dydt = [B * (self.unstalled_characteristic(x1, self.H_param, self.W_param, self.psi_c_0_param) - x3),
                (x3 - k_valve*x2**2) * B/G,
                (x1-x2)/B]
        return dydt
            
    
    def unstalled_characteristic(self, phi, H, W, psi_c_0):
            """
            It computes the unstalled characteristic of the compressor using the cubic model defined in literature

            Arguments:
                phi :  flow coefficient
                H : H parameter
                W : W parameter
                psi_c_0 : performance at zero flow coefficient
            """
            return psi_c_0 + H * (1 + 1.5*(phi/W - 1) - 0.5*(phi/W -1)**3)
    
    def plotTemporalEvolutionGreitzer(self, save_filename):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        fig, axes = plt.subplots(3,1, figsize=(10,7))
        axes[0].set_ylabel(r'$\Phi_{c}$')
        axes[0].plot(self.xi, self.solutionGreitzer[:,0])
        axes[1].set_ylabel(r'$\Phi_{t}$')
        axes[1].plot(self.xi, self.solutionGreitzer[:,1])
        axes[2].set_ylabel(r'$\Psi_p$')
        axes[2].plot(self.xi, self.solutionGreitzer[:,2])
        axes[2].set_xlabel(r'$\xi $')
        for axx in axes:
            axx.grid(alpha=grid_opacity)
        for axx in axes[0:-1]:
            axx.set_xticklabels([])
        if save_filename is not None:
            plt.savefig(PICS_FOLDER+'/'+save_filename+'_temporal_evolution.pdf', bbox_inches='tight')
    
    
    def plotTrajectoryGreitzer(self, save_filename):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        plt.figure()
        plt.plot(self.phi, self.psi_c,linewidth=2.0,label='Compressor')
        plt.plot(self.phi, self.psi_v,linewidth=2.0,label='Throttle')
        plt.plot(self.solutionGreitzer[:,0], self.solutionGreitzer[:,2],'--k',linewidth=1.0, label = 'Transient')
        plt.plot(self.solutionGreitzer[0,0], self.solutionGreitzer[0,2], 'ko')
        plt.ylabel(r'$\Psi$')
        plt.xlabel(r'$\Phi_c$')
        plt.legend()
        plt.grid(alpha=grid_opacity)
        if save_filename is not None:
            plt.savefig(PICS_FOLDER+'/'+save_filename+'_trajectory.pdf', bbox_inches='tight')
    
    
    def solveMooreGreitzerSystem(self):
        self.H_param, self.W_param, self.psi_c_0_param = self.config.get_unstalled_characteristic_params()
        self.phi = np.linspace(0,1,1000)
        self.psi_c = self.unstalled_characteristic(self.phi, self.H_param, self.W_param, self.psi_c_0_param)
        
        k_valve = self.config.get_valve_coefficient()
        self.psi_v = k_valve*self.phi**2
        
        def func_work_coefficient(phi):
            """
            Function needed by fsolve in order to find the intersection between unstalled
            compressor curve and throttle line
            """
            return self.unstalled_characteristic(phi, self.H_param, self.W_param, self.psi_c_0_param) - k_valve*phi**2

        initial_phi_guess = self.phi.min() + (self.phi.max()-self.phi.min())*0.5 
        phi_eq = fsolve(func_work_coefficient,initial_phi_guess) 
        psi_eq = k_valve*phi_eq**2 
        
        #time span
        t = np.linspace(0,self.config.get_max_time(),2500)
        
        #parameters
        self.a = self.config.get_sound_speed()
        self.Ac = self.config.get_inlet_duct_area()
        self.Vp = self.config.get_plenum_volume()
        self.Lc = self.config.get_inlet_duct_length()
        self.xi = t*self.a*np.sqrt(self.Ac/(self.Vp*self.Lc))
        self.aLag = self.config.get_moore_greitzer_time_lag()
        self.mMoore = self.config.get_moore_greitzer_m_param()
        
        #initial conditions
        perturbation = 1e-2 # work and flow coeff initial perturbations
        J_0 = 1 # initial rotating perturbation
        y0 = [psi_eq[0]*(1-perturbation), 
            phi_eq[0]*(1+perturbation), 
            J_0] 
        
        self.B_system, self.G_system = self.compute_B_G_params()
        
        self.solutionMooreGreitzer = odeint(self.mooreGreitzer3DSystem, y0, self.xi, args=(self.B_system, self.G_system, k_valve))


    def mooreGreitzer3DSystem(self, y, xi, B, G, k_valve):
        """
        Defines the differential equations for the Greitzer model
        (as found in the thesis of Sündstrom).

        Arguments:
            y :  vector of the state variables
            xi : non dimensional time
            B :  B parameter
            k_valve :  throttle line coefficient
            W : W parameter of cubic shape of compressor 
            H : H parameter of cubic shape of compressor
            psi_c_0 : work coefficient at zero flow rate parameter of cubic shape
            a : reciprocal time lag
            m : duct parameter, between 1 and 2
            lc : non dimensional compressor length
            
        State variables:
            x1 : total to static work coefficient \Psi
            x2 : azimuthally averaged flow coefficient \Phi
            x3 : squared amplitude of rotating stall cell J
        """
        x1, x2, x3 = y
        dydt = [self.W_param / (4 * self.Lc * B ** 2) * (x2 / self.W_param - (1 / self.W_param) * np.sqrt(x1 / k_valve)),
                (self.H_param / self.Lc) * (-(x1 - self.psi_c_0_param) / self.H_param + 1 + 1.5 * (x2 / self.W_param - 1) * (1 - 0.5 * x3) - 0.5 * (x2 / self.W_param - 1) ** 3),
                x3 * (1 - (x2 / self.W_param - 1) ** 2 - 0.25 * x3) * (3 * self.aLag * self.H_param) / (self.W_param * (1 + self.mMoore * self.aLag))]
        return dydt
    
    
    def plotTemporalEvolutionMooreGreitzer(self, save_filename):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        fig, axes = plt.subplots(3,1, figsize=(10,7))
        axes[0].set_ylabel(r'$\Psi$')
        axes[0].plot(self.xi, self.solutionMooreGreitzer[:, 0])
        axes[0].grid(alpha=0.3)
        axes[1].set_ylabel(r'$\Phi$')
        axes[1].plot(self.xi, self.solutionMooreGreitzer[:, 1])
        axes[1].grid(alpha=0.3)
        axes[2].set_ylabel(r'$J$')
        axes[2].plot(self.xi, self.solutionMooreGreitzer[:, 2])
        axes[2].set_xlabel(r'$\xi $')
        axes[2].grid(alpha=0.3)
        for ax in axes[0:-1]:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for axx in axes:
            axx.grid(alpha=grid_opacity)
        for axx in axes[0:-1]:
            axx.set_xticklabels([])
        if save_filename is not None:
            plt.savefig(PICS_FOLDER+'/'+save_filename+'_temporal_evolution.pdf', bbox_inches='tight')
    
    
    def plotTrajectoryMooreGreitzer(self, save_filename):
        os.makedirs(PICS_FOLDER, exist_ok=True)
        
        plt.figure()
        plt.plot(self.phi, self.psi_c,linewidth=2.0,label='Compressor')
        plt.plot(self.phi, self.psi_v,linewidth=2.0,label='Throttle')
        plt.plot(self.solutionMooreGreitzer[:,1], self.solutionMooreGreitzer[:,0],'--k',linewidth=1.0, label = 'Transient')
        plt.plot(self.solutionMooreGreitzer[0,1], self.solutionMooreGreitzer[0,0], 'ko')
        plt.ylabel(r'$\Psi$')
        plt.xlabel(r'$\Phi_c$')
        plt.legend()
        plt.grid(alpha=grid_opacity)
        if save_filename is not None:
            plt.savefig(PICS_FOLDER+'/'+save_filename+'_trajectory.pdf', bbox_inches='tight')