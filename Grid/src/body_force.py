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

    def __init__(self, config, iblade):
        self.config = config
        self.iblade = iblade
        self.bodyForceFields = {} # contaning the body force fields
        self.meridionalFields = {} # contaning all the meridional fields

        
        
    
    def InterpolateCircumferentialAveragedFields(self, zgrid, rgrid):
        """Interpolate the circumferentially averaged fields on the grid used for body force simulations

        Args:
            zgrid (np.nadarray): 2d array of axial grid coords used in the body force simulation
            rgrid (np.nadarray): 2d array of radial grid coords used in the body force simulation
        """
        
        self.axialGrid = zgrid
        self.radialGrid = rgrid   
        
        folderpath = self.config.get_circumferential_average_folder_path()
        averageType = self.config.get_circumferential_average_type()
        print("Circufemferential average type: " + averageType)
        
        if averageType.lower() == 'raw':
            filepath = os.path.join(folderpath, 'meridionalFields_rawAvg.pik')
        elif averageType.lower() == 'density':
            filepath = os.path.join(folderpath, 'meridionalFields_densityAvg.pik')
        elif averageType.lower() == 'axial_momentum':
            filepath = os.path.join(folderpath, 'meridionalFields_axialMomentumAvg.pik')
        elif averageType.lower() == 'mass_flow':
            filepath = os.path.join(folderpath, 'meridionalFields_massFlowAvg.pik')
        else:
            raise ValueError('Not valid average type')
        
        solverType = self.config.get_bladed_CFD_solver_type()
        dataset = self.preprocessProcessParaviewDataset(filepath=filepath, solver_type=solverType)
        
        # interpolate the dataset on the body force grid
        for key in dataset.keys():
            if key!='Axial_Coordinate' and key!='Radial_Coordinate':
                self.meridionalFields[key] = robust_griddata_interpolation_with_linear_filler(dataset['Axial_Coordinate'], 
                                                                                              dataset['Radial_Coordinate'], 
                                                                                              dataset[key], 
                                                                                              self.axialGrid, self.radialGrid)
        
        # now also update the grid in the dataset to be same of the body force
        self.meridionalFields['Axial_Coordinate'] = self.axialGrid
        self.meridionalFields['Radial_Coordinate'] = self.radialGrid
            
    
    def preprocessProcessParaviewDataset(self, filepath, solver_type, CP=1005, R=287, TREF=288.15, PREF=101300):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Marble Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force ion.
        """
        
        available_solver_names = ['su2', 'luminary']
        if solver_type.lower() not in available_solver_names:
            raise ValueError('Not valid solver type')
        print('Solver type: %s' % solver_type)

        # corresponding variable names between the CFD solver and Unsflow
        variablesNames = self.get_CFD_variable_names(solver_type)
        
        with open(filepath, 'rb') as file:
            averagedDataset = pickle.load(file)
            
        meridionalFields = {}
        for key in variablesNames.keys():
            meridionalFields[key] = averagedDataset[variablesNames[key]]
        
        meridionalFields['Entropy'] = CP*np.log(meridionalFields['Temperature']/TREF)-R*np.log(meridionalFields['Pressure']/PREF)
        meridionalFields['dEntropy_dz'], meridionalFields['dEntropy_dr'] = compute_gradient_least_square(meridionalFields['Axial_Coordinate'], meridionalFields['Radial_Coordinate'], meridionalFields['Entropy'])

        meridionalFields['Velocity_Meridional'] = np.sqrt(meridionalFields['Velocity_Axial']**2 + meridionalFields['Velocity_Radial']**2)
        
        # for the relative flow angle, use of simple arctan because we want it to be defined between [-pi/2,pi/2], where positive is the positive theta direction
        meridionalFields['Relative_Flow_Angle_arctan2'] = np.arctan2(meridionalFields['Velocity_Tangential_Relative'], meridionalFields['Velocity_Axial'])
        meridionalFields['Relative_Flow_Angle'] = np.arctan(meridionalFields['Velocity_Tangential_Relative']/ meridionalFields['Velocity_Axial'])
        
        meridionalFields['drUtheta_dz'], meridionalFields['drUtheta_dr'] = compute_gradient_least_square(meridionalFields['Axial_Coordinate'], meridionalFields['Radial_Coordinate'], meridionalFields['Velocity_Tangential']*meridionalFields['Radial_Coordinate'])

        # compute the 3D relative flow vector, which is generalized to radial geometries. Angle between relative velocity vector and meridional velocity vector
        meridionalFields['Relative_Flow_Angle_3D'] = np.zeros_like(meridionalFields['Relative_Flow_Angle'])
        ni,nj = meridionalFields['Relative_Flow_Angle_3D'].shape
        for i in range(ni):
            for j in range(nj):
                velAx = meridionalFields['Velocity_Axial'][i,j]
                velRad = meridionalFields['Velocity_Radial'][i,j]
                velTanRel = meridionalFields['Velocity_Tangential_Relative'][i,j]
                relVelVector = np.array([velAx, velRad, velTanRel])
                meridionalVel = np.array([velAx, velRad, 0])
                
                # need to distinguish between positive and negative angles, since arccos gives angle between [0,pi] by reference, but we want between [-pi/2, pi/2]
                if velTanRel>=0:
                    meridionalFields['Relative_Flow_Angle_3D'][i,j] = np.arccos((np.dot(relVelVector, meridionalVel)) / (np.linalg.norm(relVelVector) * np.linalg.norm(meridionalVel)))
                else:
                    meridionalFields['Relative_Flow_Angle_3D'][i,j] = -np.arccos((np.dot(relVelVector, meridionalVel)) / (np.linalg.norm(relVelVector) * np.linalg.norm(meridionalVel)))
                
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
            names['Pressure'] = 'Pressure'
            names['Temperature'] = 'Temperature'
        
        elif solver_type.lower() == 'luminary':
            names['Density'] = 'Density (kg/m³)'
            names['Pressure'] = 'Absolute Pressure (Pa)'
            names['Temperature'] = 'Temperature (K)'
        
        names['Mach'] = 'Mach'
        names['Axial_Coordinate'] = 'Axial_Coordinate'
        names['Radial_Coordinate'] = 'Radial_Coordinate'
        names['Velocity_Axial'] = 'Velocity_Axial'
        names['Velocity_Tangential'] = 'Velocity_Tangential'
        names['Velocity_Tangential_Relative'] = 'Velocity_Tangential_Relative'
        names['Velocity_Radial'] = 'Velocity_Radial'
        
        if self.config.get_body_force_extraction_method()=='kiwada':
            names['Kiwada_Term_A1'] = 'Kiwada_Term_A1'
            names['Kiwada_Term_A2'] = 'Kiwada_Term_A2'
            names['Kiwada_Term_R2'] = 'Kiwada_Term_R2'
            names['Kiwada_Term_R3'] = 'Kiwada_Term_R3'
            names['Kiwada_Term_T1'] = 'Kiwada_Term_T1'
            names['Kiwada_Term_T2'] = 'Kiwada_Term_T2'
        
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
    
    
    def PlotCalibrationCoefficients(self, save_filename):
        labels = {}
        for key in self.calibrationCoefficients.keys():
            if key.lower()!='model':
                contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], self.calibrationCoefficients[key], 
                                key, save_filename=save_filename + '_' + key, folder_name=self.config.get_pictures_folder_path())
    
    
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
            elif key=='Tau_RR':
                labels[key] = r'$\tau_{rr}^{R} \ \rm{[Pa]}$' 
            elif key=='Tau_TT':
                labels[key] = r'$\tau_{\theta \theta}^{R} \ \rm{[Pa]}$' 
            elif key=='Tau_ZZ':
                labels[key] = r'$\tau_{zz}^{R} \ \rm{[Pa]}$' 
            elif key=='Tau_RT':
                labels[key] = r'$\tau_{r \theta}^{R} \ \rm{[Pa]}$' 
            elif key=='Tau_RZ':
                labels[key] = r'$\tau_{r z}^{R} \ \rm{[Pa]}$' 
            elif key=='Tau_TZ':
                labels[key] = r'$\tau_{\theta z}^{R} \ \rm{[Pa]}$' 
            else:
                labels[key] = key
            
        for name, field in self.meridionalFields.items():
            contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, 
                             labels[name], save_filename=save_filename + '_' + name, folder_name=self.config.get_pictures_folder_path())
    
    
    def computeLossVersor(self, velRad, relVelTang, velAx):
        """Computhe the array (ni,nj,3) of loss component versors, where the components are ordered as [axial, radial, tangential]

        Args:
            velRad (np.array 2D): radial velocity
            relVelTang (np.array 2D): relative tang. velocity
            velAx (np.array 2D): axial velocity
        """
        ni,nj = velRad.shape        
        lossVersor = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                relativeVelocity = np.array([velAx[i,j],
                                             velRad[i,j],
                                             relVelTang[i,j]])
                lossVersor[i,j] = -relativeVelocity/np.linalg.norm(relativeVelocity)
        return lossVersor
    
    
    def computeTurningVersor(self, velRad, relVelTang, velAx,  n_camber_r, n_camber_t, n_camber_z):
        """Computhe the array (ni,nj,3) of turning component versors, where the components are ordered as [axial, radial, tangential]

        Args:
            velRad (np.array 2D): radial velocity
            relVelTang (np.array 2D): relative tang. velocity
            velAx (np.array 2D): axial velocity
            n_camber_r (np.array 2D): camber versor radial component
            n_camber_t (np.array 2D): camber versor tangential component
            n_camber_z (np.array 2D): camber versor axial component
        """
        ni,nj = velRad.shape        
        turnVersor = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                relativeVelocity = np.array([velAx[i,j],
                                             velRad[i,j],
                                             relVelTang[i,j]])
                
                camberNormal = np.array([n_camber_z[i,j], 
                                         n_camber_r[i,j], 
                                         n_camber_t[i,j]])
                
                turnVersor[i,j] = self.computeInviscidForceDirection(relativeVelocity, camberNormal)
        return turnVersor
    
    
    def ComputeBodyForceMarble(self, n_camber_z, n_camber_r, n_camber_t):
        """
        Compute the body force density, using the marble thermodynamic approach based on the circumferentially averaged flow field
        """
        
        # compute the basic ingredients from Marble decomposition
        self.bodyForceFields['Force_Viscous'] = self.ComputeLossForceMarble()
        self.bodyForceFields['Force_Tangential'] = self.ComputeTangentialForceMarble()
        
        # loss component versor, opposite to the relative velocity
        lossVersor = self.computeLossVersor(self.meridionalFields['Velocity_Radial'], 
                                            self.meridionalFields['Velocity_Tangential_Relative'], 
                                            self.meridionalFields['Velocity_Axial'])
        
        # turning component versor, orthogonal to relative velocity
        turnVersor = self.computeTurningVersor(self.meridionalFields['Velocity_Radial'], 
                                               self.meridionalFields['Velocity_Tangential_Relative'], 
                                               self.meridionalFields['Velocity_Axial'], 
                                               n_camber_r, n_camber_t, n_camber_z)

        # viscous force cartesian components
        self.bodyForceFields['Force_Viscous_Radial'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,1]
        self.bodyForceFields['Force_Viscous_Tangential'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,2]
        self.bodyForceFields['Force_Viscous_Axial'] = self.bodyForceFields['Force_Viscous']*lossVersor[:,:,0]
        
        self.bodyForceFields['Force_Inviscid_Tangential'] = self.bodyForceFields['Force_Tangential']-self.bodyForceFields['Force_Viscous_Tangential']
        self.bodyForceFields['Force_Inviscid'] = np.abs(self.bodyForceFields['Force_Inviscid_Tangential'] / turnVersor[:,:,2])
        self.bodyForceFields['Force_Inviscid_Radial'] = self.bodyForceFields['Force_Inviscid'] * turnVersor[:,:,1]
        self.bodyForceFields['Force_Inviscid_Axial'] = self.bodyForceFields['Force_Inviscid'] * turnVersor[:,:,0]
        
        self.bodyForceFields['Force_Radial'] = self.bodyForceFields['Force_Viscous_Radial']+self.bodyForceFields['Force_Inviscid_Radial']
        self.bodyForceFields['Force_Axial'] = self.bodyForceFields['Force_Viscous_Axial']+self.bodyForceFields['Force_Inviscid_Axial']
        
        # make plots to the check the correct orientation
        offset = 10
        for key in self.bodyForceFields.keys():
            contour_template(self.axialGrid[:,offset:-1-offset], self.radialGrid[:,offset:-1-offset], self.bodyForceFields[key][:,offset:-1-offset], key, save_filename=key, folder_name='forceDirections')

        print()
    
    def ComputeBodyForceKiwada(self, blockage):
        """Use the Kiwada Blade Force Average to extract the BF
        """
        # Compute the terms marked in red in the Kiwada BFA section of thollet thesis
        A1 = self.meridionalFields['Kiwada_Term_A1'] 
        A2 = self.meridionalFields['Kiwada_Term_A2']
        R2 = self.meridionalFields['Kiwada_Term_R2']
        R3 = self.meridionalFields['Kiwada_Term_R3']
        T1 = self.meridionalFields['Kiwada_Term_T1']
        T2 = self.meridionalFields['Kiwada_Term_T2'] 
        
        Z = self.meridionalFields['Axial_Coordinate']
        R = self.meridionalFields['Radial_Coordinate']
        dbdz, dbdr = compute_gradient_least_square(Z, R, blockage)

        contour_template(Z, R, blockage, r'$b$')
        contour_template(Z, R, dbdz, r'$\partial_z b$')
        contour_template(Z, R, dbdr, r'$\partial_r b$')

        # axial equation
        dA1dz = compute_gradient_least_square(Z, R, blockage * A1)[0]
        dA2dr = compute_gradient_least_square(Z, R, blockage * R * A2)[1]
        self.bodyForceFields['Force_Axial'] = 1/blockage * dA1dz + 1/blockage/R*dA2dr
        # contour_template(Z, R, self.bodyForceFields['Force_Axial'], r'$Fx$', vmin=0)

        dR1dz = compute_gradient_least_square(Z, R, blockage * A2)[0]
        dR2dr = compute_gradient_least_square(Z, R, blockage * R * R2)[1]
        self.bodyForceFields['Force_Radial'] = 1/blockage*dR1dz+1/blockage/R*dR2dr-R3/R
        # contour_template(Z, R, self.bodyForceFields['Force_Radial'], r'$Fr$')

        dT1dz = compute_gradient_least_square(Z, R, blockage * T1)[0]
        dT2dr = compute_gradient_least_square(Z, R, blockage * R * T2)[1]
        self.bodyForceFields['Force_Tangential'] = 1/blockage*dT1dz + 1/blockage/R*dT2dr + T2/R
        # contour_template(Z, R, self.bodyForceFields['Force_Tangential'], r'$Ft$', vmin=0)

        globalForceMagnitude = np.sqrt(self.bodyForceFields['Force_Radial']**2+
                                       self.bodyForceFields['Force_Tangential']**2+
                                       self.bodyForceFields['Force_Axial']**2)

        ni,nj = R.shape
        self.bodyForceFields['Force_Viscous_Radial'] = np.zeros((ni,nj))
        self.bodyForceFields['Force_Viscous_Tangential'] = np.zeros((ni,nj))
        self.bodyForceFields['Force_Viscous_Axial'] = np.zeros((ni,nj))
        self.bodyForceFields['Force_Viscous'] = np.zeros((ni,nj))
        for i in range(ni):
            for j in range(nj):
                relVelocity = np.array([self.meridionalFields['Velocity_Radial'][i,j],
                                        self.meridionalFields['Velocity_Tangential_Relative'][i,j],
                                        self.meridionalFields['Velocity_Axial'][i,j]])
                globalForce = np.array([self.bodyForceFields['Force_Radial'][i,j],
                                        self.bodyForceFields['Force_Tangential'][i,j],
                                        self.bodyForceFields['Force_Axial'][i,j]])
                
                lossVersor = -relVelocity/np.linalg.norm(relVelocity)
                self.bodyForceFields['Force_Viscous'][i,j] = np.dot(globalForce, lossVersor)
                self.bodyForceFields['Force_Viscous_Radial'][i,j] = self.bodyForceFields['Force_Viscous'][i,j]*lossVersor[0]
                self.bodyForceFields['Force_Viscous_Tangential'][i,j] = self.bodyForceFields['Force_Viscous'][i,j]*lossVersor[1]
                self.bodyForceFields['Force_Viscous_Axial'][i,j] = self.bodyForceFields['Force_Viscous'][i,j]*lossVersor[2]
                
        self.bodyForceFields['Force_Inviscid_Radial'] = self.bodyForceFields['Force_Radial']-self.bodyForceFields['Force_Viscous_Radial']
        self.bodyForceFields['Force_Inviscid_Axial'] = self.bodyForceFields['Force_Axial']-self.bodyForceFields['Force_Viscous_Axial']
        self.bodyForceFields['Force_Inviscid_Tangential'] = self.bodyForceFields['Force_Tangential']-self.bodyForceFields['Force_Viscous_Tangential']
        self.bodyForceFields['Force_Inviscid'] = np.sqrt(self.bodyForceFields['Force_Inviscid_Axial']**2+
                                                         self.bodyForceFields['Force_Inviscid_Radial']**2+
                                                         self.bodyForceFields['Force_Inviscid_Tangential']**2)
        
        
    def ComputeLossForceMarble(self, method='distributed'):
        """Compute the loss force component according to Marble method, distributing linearly the loss from leading to trailing edge

        Returns:
            np.ndarray: 2D array of the loss force
        """
        force = np.zeros_like(self.meridionalFields['Axial_Coordinate'])
        temperature = self.meridionalFields['Temperature']
        relativeVelocity = sqrt(self.meridionalFields['Velocity_Axial']**2 + self.meridionalFields['Velocity_Radial']**2 + self.meridionalFields['Velocity_Tangential_Relative']**2)
        meridionalVelocity = np.sqrt(self.meridionalFields['Velocity_Axial']**2 + self.meridionalFields['Velocity_Radial']**2)
        streamCoords = compute_meridional_streamwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'])
        normalizedSpan = compute_meridional_spanwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], normalize=True)

        entropyMeridionalDerivative = np.zeros_like(self.meridionalFields['Axial_Coordinate'])
        
        # first implementation
        offset = 1
        
        if method=='distributed':
            for j in range(force.shape[1]):
                deltaEntropy = self.meridionalFields['Entropy'][-offset,j]-self.meridionalFields['Entropy'][offset,j]
                deltaLength = streamCoords[-offset,j]-streamCoords[offset,j]
                entropyMeridionalDerivative[:,j] = deltaEntropy/deltaLength
                force[:,j] = temperature[:,j]*meridionalVelocity[:,j]/relativeVelocity[:,j]*deltaEntropy/deltaLength
        else:
            ds_dz, ds_dr = compute_gradient_least_square(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], self.meridionalFields['Entropy'])
            entropyMeridionalDerivative = ds_dz*self.meridionalFields['Velocity_Axial']+ds_dr*self.meridionalFields['Velocity_Radial']
            force = temperature/relativeVelocity*entropyMeridionalDerivative
        
        self.meridionalFields['EntropyDerivative'] = entropyMeridionalDerivative
        
        
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
        streamCoords = compute_meridional_streamwise_coordinates(zgrid, rgrid)
        offset = 1

        # first implementation type based on meridional fields extracted from CFD
        if method=='local':
            drut_dz, drut_dr = self.meridionalFields['drUtheta_dz'], self.meridionalFields['drUtheta_dr']
            force = (drut_dz*self.meridionalFields['Velocity_Axial']+drut_dr*self.meridionalFields['Velocity_Radial']) / rgrid
        
        elif method=='distributed':
            force = np.zeros_like(meridionalVelocity)
            for j in range(meridionalVelocity.shape[1]):
                deltaForce = rgrid[-offset,j]*tangentialVelocity[-offset,j] - rgrid[offset,j]*tangentialVelocity[offset,j]
                deltaLength = streamCoords[-offset,j]-streamCoords[offset,j]
                force[:,j] = meridionalVelocity[:,j]/rgrid[:,j]*deltaForce/deltaLength
        else:
            raise ValueError('Method unknown')
        
        self.meridionalFields['AngularMomentumDerivative'] = force*rgrid/meridionalVelocity
                
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
            self.bodyForceFields[key] = zeroOrderExtrapolation(self.bodyForceFields[key], spanLength, self.config.hub_shroud_body_force_extrapolation_span_extent())
        
        for key in self.meridionalFields.keys():
            if key!='Radial_Coordinate' and key!="Axial_Coordinate":
                self.meridionalFields[key] = zeroOrderExtrapolation(self.meridionalFields[key], spanLength, self.config.hub_shroud_body_force_extrapolation_span_extent())

                
    def SaveOutput(self):
        name = self.config.get_body_force_blade_name() + '.pik'
        with open(name, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved body force fields in file: {name}!")
    
    
    def ComputeCalibrationCoefficients(self, calibration_method, n_camber_z, n_camber_r, n_camber_t):
        metal_angle = np.arctan(np.sqrt(n_camber_r**2+n_camber_z**2) / n_camber_t)
        
        if calibration_method.lower()=='lift/drag':
            self.ComputeLiftDragCalibrationCoefficients(metal_angle)
    
    
    def ComputeLiftDragCalibrationCoefficients(self, metal_angle):
        """Computhe the lift/drag calibration coefficients

        Args:
            metal_angle (np.array 2D): metal angle of the blade
        """
        nBlades = self.config.get_blades_number()[self.iblade]
        circumferentialPitch = 2 * np.pi * self.radialGrid / nBlades
        
        contour_template(self.axialGrid, self.radialGrid, metal_angle*180/np.pi, 'metal_angle [deg]')
        h_parameter = circumferentialPitch * np.cos(metal_angle)
        streamLength = compute_meridional_streamwise_coordinates(self.axialGrid, self.radialGrid)
        solidity = np.zeros_like(streamLength)
        for i in range(solidity.shape[0]):
            solidity[i,:] = streamLength[-1,:] / circumferentialPitch[i,:]
        
        contour_template(self.axialGrid, self.radialGrid, circumferentialPitch, 'circumferential_pitch')
        contour_template(self.axialGrid, self.radialGrid, metal_angle*180/np.pi, 'metal_angle [deg]')
        contour_template(self.axialGrid, self.radialGrid, h_parameter, 'h_parameter')
        contour_template(self.axialGrid, self.radialGrid, solidity, 'solidity')
        contour_template(self.axialGrid, self.radialGrid, self.meridionalFields["Relative_Flow_Angle_3D"]*180/np.pi, 'relative flow angle [deg]')
        
        beta_flow = self.meridionalFields['Relative_Flow_Angle_3D']
        
        relVelMag = np.sqrt(self.meridionalFields["Velocity_Axial"]**2 + self.meridionalFields["Velocity_Radial"]**2 + self.meridionalFields["Velocity_Tangential_Relative"]**2)
        beta_0 = beta_flow - self.bodyForceFields["Force_Inviscid"] * h_parameter / (2*np.pi*solidity*relVelMag**2)
        kp_etaMax = h_parameter * self.bodyForceFields["Force_Viscous"] / (relVelMag**2)
        beta_etaMax = beta_flow
        
        contour_template(self.axialGrid, self.radialGrid, beta_0*180/np.pi, 'beta_0 [deg]')
        contour_template(self.axialGrid, self.radialGrid, kp_etaMax, 'kp_etaMax')
        contour_template(self.axialGrid, self.radialGrid, beta_etaMax*180/np.pi, 'beta_etaMax [deg]')
        
        self.calibrationCoefficients = {}
        # self.calibrationCoefficients["Model"] = 'lift/drag'
        self.calibrationCoefficients["beta_0"] = beta_0
        self.calibrationCoefficients["kp_etaMax"] = kp_etaMax
        self.calibrationCoefficients["beta_etaMax"] = beta_etaMax
        self.calibrationCoefficients["solidity"] = solidity
        self.calibrationCoefficients["h_parameter"] = h_parameter # try to understand if it makes sense, or if it is better to use the circumferential pitch
        
        deviation = beta_flow - metal_angle
        contour_template(self.axialGrid, self.radialGrid, deviation*180/np.pi, 'deviation [deg]', vmin=-5, vmax=5)
        self.calibrationCoefficients["kn_turning"] = self.bodyForceFields["Force_Inviscid"] * h_parameter / (relVelMag**2 * np.abs(deviation))
        contour_template(self.axialGrid, self.radialGrid, self.calibrationCoefficients["kn_turning"], 'kn_turning')
    
    
    def computeInviscidForceDirection(self, w, n, tangentialComponentPositive=True):
        w += 1e-9 # to cope with anormal cases
        w_dir = w/np.linalg.norm(w) 
        n_dir = n/np.linalg.norm(n)
        w_ax = w_dir[0]
        w_rad = w_dir[1]
        w_tan = w_dir[2]
        n_ax = n_dir[0]
        n_rad = n_dir[1]
        n_tan = n_dir[2]
        A = w_tan**2 + w_ax**2
        B = 2 * w_rad * w_ax * n_rad
        C = (w_tan**2 * n_rad**2) + (w_rad**2 * n_rad**2) - w_tan**2 
        deltaEquation = B**2-4*A*C
        
        if deltaEquation < 0:
            print('Delta quadratic equation negative')
        
        fAxial_1 = (-B + np.sqrt(deltaEquation))/2/A
        fAxial_2 = (-B - np.sqrt(deltaEquation))/2/A
        
        def compute_tangential(fax):
            ftan = (-w_ax*fax - w_rad*n_rad)/w_tan
            return ftan
        
        fTangential_1 = compute_tangential(fAxial_1)
        fTangential_2 = compute_tangential(fAxial_2)
        
        fRadial_1 = n_rad
        fRadial_2 = n_rad
        
        fn_versor_1 = np.array([fAxial_1, fRadial_1, fTangential_1])
        fn_versor_2 = np.array([fAxial_2, fRadial_2, fTangential_2])
        
        if np.dot(fn_versor_1, n) > 0:
            fn_dir = fn_versor_1
        else:
            fn_dir = fn_versor_2
        
        # choose the versor based on the tangential component being positive or negative
        # if tangentialComponentPositive:
        #     if fn_versor_1[2]>=0:
        #         fn_dir = fn_versor_1
        #     else:
        #         fn_dir = fn_versor_2
        # else:
        #     if fn_versor_1[2]<=0:
        #         fn_dir = fn_versor_1
        #     else:
        #         fn_dir = fn_versor_2
        
        if any(math.isnan(x) for x in fn_dir):
            print("NaN found during calculation of inviscid force direction. Radial component set to zero.")
            fn_dir = np.array([-w_dir[2], 0, w_dir[0]])

        return fn_dir