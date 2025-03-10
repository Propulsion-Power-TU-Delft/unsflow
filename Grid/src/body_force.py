from Grid.src.config import Config
import os
import numpy as np
from numpy import sin, cos, tan, arctan2, pi, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from Grid.src.functions import *
import pickle


class BodyForce:
    """
    class that stores the information regarding the blade topology.
    """

    def __init__(self, config):
        self.config = config
        self.bodyForceFields = {} # contaning the body force fields
        self.meridionalFields = {} # contaning all the meridional fields
    
    def ComputeCircumferentialAveragedFields(self, zgrid, rgrid):
        """Interpolate the circumferentially averaged fields on the grid used for body force simulations

        Args:
            zgrid (np.nadarray): 2d array of axial grid coords used in the body force simulation
            rgrid (np.nadarray): 2d array of radial grid coords used in the body force simulation
        """
        
        self.axialGrid = zgrid
        self.radialGrid = rgrid   
        
        filepath = self.config.get_circumferential_average_fields_path()
        solverType = self.config.get_bladed_CFD_solver_type()
        self.meridionalFields = self.ProcessParaviewDataset(filepath=filepath, solver_type=solverType)
        
        for key in self.meridionalFields.keys():
            if key!='Axial_Coordinate' and key!='Radial_Coordinate':
                self.meridionalFields[key] = griddata_interpolation_with_nearest_filler(self.meridionalFields['Axial_Coordinate'], 
                                                                                        self.meridionalFields['Radial_Coordinate'], 
                                                                                        self.meridionalFields[key], 
                                                                                        self.axialGrid, self.radialGrid)
        self.meridionalFields['Axial_Coordinate'] = self.axialGrid
        self.meridionalFields['Radial_Coordinate'] = self.radialGrid
            
    
    def ProcessParaviewDataset(self, filepath, solver_type, CP=1005, R=287, TREF=288.15, PREF=101300):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Marble Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force extraction.
        """
        
        available_solver_names = ['su2', 'luminary']
        if solver_type.lower() not in available_solver_names:
            raise ValueError('Not valid solver type')
        print('Solver type: %s' % solver_type)

        variablesNames = self.get_CFD_variable_names(solver_type)
        
        with open(filepath, 'rb') as file:
            averagedDataset = pickle.load(file)
            
        meridionalFields = {}
        for key in variablesNames.keys():
            meridionalFields[key] = averagedDataset[variablesNames[key]]
        
        meridionalFields['Entropy'] = CP*np.log(meridionalFields['Temperature']/TREF)-R*np.log(meridionalFields['Pressure']/PREF)
        meridionalFields['Relative_Flow_Angle'] = np.arctan2(meridionalFields['Velocity_Tangential_Relative'], meridionalFields['Velocity_Axial'])
            
        return meridionalFields
    
    
    def get_CFD_variable_names(self, solver_type):
        """Return the dictionary with all the correct variable names depending on the solver employed for the CFD. These will be the variable stored in the bfm object

        Args:
            solver_type (string): Luminary or SU2

        Returns:
            dict: dictionary of names
        """
        names = {}

        if solver_type.lower() == 'su2':
            names['Density'] = 'Density'
            # names['Velocity_X'] = 'Velocity_0'
            # names['Velocity_Y'] = 'Velocity_1'
            # names['Velocity_Z'] = 'Velocity_2'
            names['Pressure'] = 'Pressure'
            names['Temperature'] = 'Temperature'
            # names['Grid_Velocity_X'] = 'Grid_Velocity_0'
            # names['Grid_Velocity_Y'] = 'Grid_Velocity_1'
            # names['Grid_Velocity_Z'] = 'Grid_Velocity_2'
        
        elif solver_type.lower() == 'luminary':
            names['Density'] = 'Density (kg/m³)'
            # names['Velocity_X'] = 'Velocity (m/s)_0'
            # names['Velocity_Y'] = 'Velocity (m/s)_1'
            # names['Velocity_Z'] = 'Velocity (m/s)_2'
            names['Pressure'] = 'Absolute Pressure (Pa)'
            names['Temperature'] = 'Temperature (K)'
            # names['Grid_Velocity_X'] = 'Grid Velocity (m/s)_0'
            # names['Grid_Velocity_Y'] = 'Grid Velocity (m/s)_1'
            # names['Grid_Velocity_Z'] = 'Grid Velocity (m/s)_2'
        
        names['Mach'] = 'Mach'
        names['Axial_Coordinate'] = 'Axial_Coordinate'
        names['Radial_Coordinate'] = 'Radial_Coordinate'
        names['Velocity_Axial'] = 'Velocity_Axial'
        names['Velocity_Tangential'] = 'Velocity_Tangential'
        names['Velocity_Tangential_Relative'] = 'Velocity_Tangential_Relative'
        names['Velocity_Radial'] = 'Velocity_Radial'
        
        return names
    
    
    def PlotMeridionalFields(self):
        for name, field in self.meridionalFields.items():
            if name != 'Axial_Coordinate' and name!= 'Radial_Coordinate':
                contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, name)
    
    def PlotBodyForceFields(self, save_filename):
        labels = {}
        for key in self.bodyForceFields.keys():
            if key=='Force_Axial':
                labels[key] = r'$f_{ax} \ \rm{[N/kg]}$'
            elif key=='Force_Radial':
                labels[key] = r'$f_{r} \ \rm{[N/kg]}$'
            elif key=='Force_Tangential':
                labels[key] = r'$f_{\theta} \ \rm{[N/kg]}$'
            elif key=='Force_Inviscid':
                labels[key] = r'$f_{n} \ \rm{[N/kg]}$'
            elif key=='Force_Viscous':
                labels[key] = r'$f_{p} \ \rm{[N/kg]}$'
            elif key=='Force_Inviscid_Axial':
                labels[key] = r'$f_{n,ax} \ \rm{[N/kg]}$'
            elif key=='Force_Inviscid_Radial':
                labels[key] = r'$f_{n,r} \ \rm{[N/kg]}$' 
            elif key=='Force_Inviscid_Tangential':
                labels[key] = r'$f_{n,\theta} \ \rm{[N/kg]}$'
            elif key=='Force_Viscous_Axial':
                labels[key] = r'$f_{p,ax} \ \rm{[N/kg]}$'
            elif key=='Force_Viscous_Radial':
                labels[key] = r'$f_{p,r} \ \rm{[N/kg]}$'
            elif key=='Force_Viscous_Tangential':
                labels[key] = r'$f_{p,\theta} \ \rm{[N/kg]}$'
            else:
                labels[key] = key
            
        for name, field in self.bodyForceFields.items():
            contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, 
                             labels[name], save_filename=save_filename + '_' + name, folder_name=self.config.get_pictures_folder_path())
    
    
    def PlotCircumferentiallyAveragedFields(self, save_filename):
        labels = {}
        for key in self.meridionalFields.keys():
            if key=='Density':
                labels[key] = r'$\rho \ \rm{[kg/m^3]}$'
            elif key=='Pressure':
                labels[key] = r'$p \ \rm{[Pa]}$'
            elif key=='Temperature':
                labels[key] = r'$T \ \rm{[K]}$'
            elif key=='Mach':
                labels[key] = r'$M \ \rm{[-]}$'
            elif key=='Velocity_Axial':
                labels[key] = r'$u_{ax} \ \rm{[m/s]}$'
            elif key=='Velocity_Tangential':
                labels[key] = r'$u_{\theta} \ \rm{[m/s]}$'
            elif key=='Velocity_Tangential_Relative':
                labels[key] = r'$w_{\theta} \ \rm{[m/s]}$'
            elif key=='Velocity_Radial':
                labels[key] = r'$u_{r} \ \rm{[m/s]}$'
            elif key=='Entropy':
                labels[key] = r'$s \ \rm{[J/kgK]}$' 
            elif key=='Relative_Flow_Angle':
                labels[key] = r'$\beta \ \rm{[rad]}$' 
            else:
                labels[key] = key
            
        for name, field in self.meridionalFields.items():
            contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, 
                             labels[name], save_filename=save_filename + '_' + name, folder_name=self.config.get_pictures_folder_path())
    
    
    def ComputeBodyForceMarble(self, leanAngle):
        """
        Compute the body force density, using the marble thermodynamic approach based on the circumferentially averaged flow field
        """
        
        self.bodyForceFields['Force_Viscous'] = self.ComputeLossForceMarble()
        self.bodyForceFields['Force_Tangential'] = self.ComputeTangentialForceMarble()

        ni,nj = self.bodyForceFields['Force_Viscous'].shape
        lossVersor = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                relativeVelocity = np.array([self.meridionalFields['Velocity_Radial'][i,j],
                                             self.meridionalFields['Velocity_Tangential_Relative'][i,j],
                                             self.meridionalFields['Velocity_Axial'][i,j]])
                lossVersor[i,j] = -relativeVelocity/np.linalg.norm(relativeVelocity)

        self.bodyForceFields['Force_Viscous_Radial'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,0]
        
        self.bodyForceFields['Force_Viscous_Tangential'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,1]
        
        self.bodyForceFields['Force_Viscous_Axial'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,2]
        
        self.bodyForceFields['Force_Inviscid_Tangential'] = self.bodyForceFields['Force_Tangential']-self.bodyForceFields['Force_Viscous_Tangential']
        
        self.bodyForceFields['Force_Inviscid_Radial'] = np.abs(self.bodyForceFields['Force_Inviscid_Tangential'])*np.tan(leanAngle)

        self.bodyForceFields['Force_Axial'] = self.bodyForceFields['Force_Viscous_Axial']-(self.bodyForceFields['Force_Tangential']-
                                                self.bodyForceFields['Force_Viscous_Tangential'])*self.meridionalFields['Velocity_Tangential_Relative']/self.meridionalFields['Velocity_Axial']
        
        self.bodyForceFields['Force_Inviscid_Axial'] = self.bodyForceFields['Force_Axial']-self.bodyForceFields['Force_Viscous_Axial']

        self.bodyForceFields['Force_Inviscid'] = np.sqrt(self.bodyForceFields['Force_Inviscid_Axial']**2+
                                                           self.bodyForceFields['Force_Inviscid_Radial']**2+
                                                           self.bodyForceFields['Force_Inviscid_Tangential']**2)
        
        self.bodyForceFields['Force_Viscous'] = np.sqrt(self.bodyForceFields['Force_Viscous_Axial']**2+
                                                        self.bodyForceFields['Force_Viscous_Radial']**2+
                                                        self.bodyForceFields['Force_Viscous_Tangential']**2)
        
        self.bodyForceFields['Force_Radial'] = self.bodyForceFields['Force_Viscous_Radial']+self.bodyForceFields['Force_Inviscid_Radial']
        
        
    
    
    def ComputeLossForceMarble(self):
        """Compute the loss force component according to Marble method, distributing linearly the loss from leading to trailing edge

        Returns:
            np.ndarray: 2D array of the loss force
        """
        force = np.zeros_like(self.meridionalFields['Axial_Coordinate'])
        temperature = self.meridionalFields['Temperature']
        relativeVelocity = sqrt(self.meridionalFields['Velocity_Axial']**2 + self.meridionalFields['Velocity_Radial']**2 + self.meridionalFields['Velocity_Tangential_Relative']**2)
        meridionalVelocity = np.sqrt(self.meridionalFields['Velocity_Axial']**2 + self.meridionalFields['Velocity_Radial']**2)
        streamLength = compute_meridional_streamwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'])

        for j in range(force.shape[1]):
            deltaEntropy = self.meridionalFields['Entropy'][-1,j]-self.meridionalFields['Entropy'][0,j]
            deltaLength = streamLength[-1,j]-streamLength[0,j]
            force[:,j] = temperature[:,j]*meridionalVelocity[:,j]/relativeVelocity[:,j]*deltaEntropy/deltaLength
        return force
    
    
    def ComputeTangentialForceMarble(self, method='local'):
        """Compute the tangential force component according to Marble method
        
        Args:
            method (str, optional): Method to compute the tangential force. Defaults to 'local' computes local gradient. If set to 'distributed' 
            the force is linearly distributed from leading to trailing edge just like the loss component

        Returns:
            np.ndarray: 2D array of the tangential force
        """
        meridionalVelocity = np.sqrt(self.meridionalFields['Velocity_Axial']**2 + self.meridionalFields['Velocity_Radial']**2)
        tangentialVelocity = self.meridionalFields['Velocity_Tangential']
        rgrid = self.meridionalFields['Radial_Coordinate']
        zgrid = self.meridionalFields['Axial_Coordinate']
        streamLength = compute_meridional_streamwise_coordinates(zgrid, rgrid)
        
        if method=='local':
            drut_dz, drut_dr = compute_gradient_least_square(zgrid, rgrid, rgrid*tangentialVelocity)
            force = 1/rgrid*(drut_dz*self.meridionalFields['Velocity_Axial']+drut_dr*self.meridionalFields['Velocity_Radial'])
        
        elif method=='distributed':
            force = np.zeros_like(meridionalVelocity)
            for j in range(meridionalVelocity.shape[1]):
                deltaForce = rgrid[-1,j]*tangentialVelocity[-1,j] - rgrid[0,j]*tangentialVelocity[0,j]
                deltaLength = streamLength[-1,j]-streamLength[0,j]
                force[:,j] = meridionalVelocity[:,j]/rgrid[:,j]*deltaForce/deltaLength
        
        else:
            raise ValueError('Method unknown')
        
        return force
    
    
    def CutBladeTip(self):
        """
        Remove every force component in the gap from the shroud described by clearance_meters
        """
        extension = self.config.cut_body_force_blade_tip_extension()
        spanLength = compute_meridional_spanwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], normalize=True)
        ni,nj = spanLength.shape
        
        for i in range(ni):
            cutIndices = np.where(spanLength[i,:]>=1-extension)
            for key in self.bodyForceFields.keys():
                self.bodyForceFields[key][i,cutIndices] = 0
    
    
    def HubShroudBodyForceExtrapolation(self):
        """
        Extrapolate the body force fields in the proximity of hub and shroud, up to a certain span extent
        """
        spanExtent = self.config.hub_shroud_body_force_extrapolation_span_extent()
        if spanExtent < 1e-3:
            return
        
        spanLength = compute_meridional_spanwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], normalize=True)
        
        def zeroOrderExtrapolation(field, spanwiseCoords, spanExtent):
            ni,nj = field.shape
            for i in range(ni):
                idx = np.where(spanwiseCoords[i,:]<spanExtent)
                field[i,idx] = field[i, idx[0][-1]+1]
                
                idx = np.where(spanwiseCoords[i,:]>1-spanExtent)
                field[i,idx] = field[i,idx[0][0]-1]                
            return field
        
        for key in self.bodyForceFields.keys():
            if key != 'Force_Viscous':
                self.bodyForceFields[key] = zeroOrderExtrapolation(self.bodyForceFields[key], spanLength, self.config.hub_shroud_body_force_extrapolation_span_extent())
                
                
    def SaveOutput(self):
        name = self.config.get_body_force_blade_name() + '.pik'
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved body force fields in file: {name}!")
        