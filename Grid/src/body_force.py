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
        self.spanwiseProfiles = []
        
        self.processSpanwiseProfiles(iblade)
    
    def processSpanwiseProfiles(self, iblade, CP=1004, R=287, TREF=300, PREF=101325):
        GAMMA = self.config.get_fluid_gamma()
        
        # read the profiles data
        spanwisePaths = self.config.get_spanwise_profiles_paths(iblade)
        for path in spanwisePaths:
            with open(path, 'rb') as f:
                self.spanwiseProfiles.append(pickle.load(f))
        
        # clean possible nan values
        for profile in self.spanwiseProfiles:
            for key in profile:
                if key!='Streamwise_Coordinate':
                    idxValid = ~np.isnan(profile[key])
                    profile[key] = profile[key][idxValid]
                else:
                    profile[key] = profile[key][idxValid]
            
            profile['NormalizedSpan'] = compute_meridional_spanwise_coordinates(profile['Axial_Coordinate'], profile['Radial_Coordinate'], normalize=True)[0,:]

            profile['AngularMomentum'] = profile['Velocity_Tangential'] * profile['Radial_Coordinate']
            
            profile['Entropy'] = CP*np.log(profile['Temperature (K)']/TREF)-R*np.log(profile['Pressure (Pa)']/PREF)
        
        # plt.figure()
        # for profile in self.spanwiseProfiles:
        #     plt.plot(profile['Entropy'], profile['NormalizedSpan'], '-.',label='Inlet')
        # plt.xlabel('entropy')
        
        # plt.figure()
        # for profile in self.spanwiseProfiles:
        #     plt.plot(profile['AngularMomentum'], profile['NormalizedSpan'], '-.',label='Inlet')
        # plt.xlabel('angular momentum')
                
        
        
        
    
    def ComputeCircumferentialAveragedFields(self, zgrid, rgrid):
        """Interpolate the circumferentially averaged fields on the grid used for body force simulations

        Args:
            zgrid (np.nadarray): 2d array of axial grid coords used in the body force simulation
            rgrid (np.nadarray): 2d array of radial grid coords used in the body force simulation
        """
        
        self.axialGrid = zgrid
        self.radialGrid = rgrid   
        
        folderpath = self.config.get_circumferential_average_folder_path()
        averageType = self.config.get_circumferential_average_type()
        
        if averageType.lower() == 'raw':
            filepath = os.path.join(folderpath, 'meridionalFields_rawAvg.pik')
        elif averageType.lower() == 'density':
            filepath = os.path.join(folderpath, 'meridionalFields_densityAvg.pik')
        elif averageType.lower() == 'axial_momentum':
            filepath = os.path.join(folderpath, 'meridionalFields_axialMomentumAvg.pik')
        else:
            raise ValueError('Not valid average type')
        
        solverType = self.config.get_bladed_CFD_solver_type()
        dataset = self.ProcessParaviewDataset(filepath=filepath, solver_type=solverType)
                
        for key in dataset.keys():
            if key!='Axial_Coordinate' and key!='Radial_Coordinate':
                # self.meridionalFields[key] = griddata_interpolation_with_linear_extrapolation(dataset['Axial_Coordinate'], 
                #                                                                               dataset['Radial_Coordinate'], 
                #                                                                               dataset[key], 
                #                                                                               self.axialGrid, self.radialGrid)
                self.meridionalFields[key] = griddata_interpolation_with_nearest_filler(dataset['Axial_Coordinate'], 
                                                                                              dataset['Radial_Coordinate'], 
                                                                                              dataset[key], 
                                                                                              self.axialGrid, self.radialGrid)
        
        self.meridionalFields['Axial_Coordinate'] = self.axialGrid
        self.meridionalFields['Radial_Coordinate'] = self.radialGrid
            
    
    def ProcessParaviewDataset(self, filepath, solver_type, CP=1005, R=287, TREF=288.15, PREF=101300):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Marble Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force ion.
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
        meridionalFields['Velocity_Meridional'] = np.sqrt(meridionalFields['Velocity_Axial']**2 + meridionalFields['Velocity_Radial']**2)
        
        # for the relative flow angle, use of simple arctan because we want it to be defined between [-pi/2,pi/2], where positive is the positive theta direction
        meridionalFields['Relative_Flow_Angle_arctan2'] = np.arctan2(meridionalFields['Velocity_Tangential_Relative'], meridionalFields['Velocity_Axial'])
        meridionalFields['Relative_Flow_Angle'] = np.arctan(meridionalFields['Velocity_Tangential_Relative']/ meridionalFields['Velocity_Axial'])
        
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
        
        # verify that the three angles are the same. If not, there is a problem somewhere
        # contour_template(meridionalFields['Axial_Coordinate'], meridionalFields['Radial_Coordinate'], meridionalFields['Relative_Flow_Angle_arctan2']*180/np.pi, 'beta arctan2', vmin=-70, vmax=25)
        # contour_template(meridionalFields['Axial_Coordinate'], meridionalFields['Radial_Coordinate'], meridionalFields['Relative_Flow_Angle']*180/np.pi, 'beta arctan', vmin=-70, vmax=25)
        contour_template(meridionalFields['Axial_Coordinate'], meridionalFields['Radial_Coordinate'], meridionalFields['Relative_Flow_Angle_3D']*180/np.pi, 'beta 3D', vmin=-70, vmax=25)
        
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
    
    
    def ComputeBodyForceMarble(self, n_camber_r):
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
        
        self.bodyForceFields['Force_Inviscid_Radial'] = np.abs(self.bodyForceFields['Force_Inviscid_Tangential'])*np.tan(n_camber_r)

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
        
        # # clip spurious values of the cartesian force
        # if not self.config.invert_axial_coordinates():
        #     clip_negative_values(self.bodyForceFields['Force_Axial'])
        # else:
        #     clip_positive_values(self.bodyForceFields['Force_Axial'])
        
        # if self.config.get_omega_shaft()[self.iblade]>0:
        #     clip_negative_values(self.bodyForceFields['Force_Tangential'])
        # elif self.config.get_omega_shaft()[self.iblade]<0:
        #     clip_positive_values(self.bodyForceFields['Force_Tangential'])
        # else:
        #     pass
    
    
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
        
        
    def ComputeLossForceMarble(self):
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
        for j in range(force.shape[1]):
            deltaEntropy = self.meridionalFields['Entropy'][-offset,j]-self.meridionalFields['Entropy'][offset,j]
            deltaLength = streamCoords[-offset,j]-streamCoords[offset,j]
            entropyMeridionalDerivative[:,j] = deltaEntropy/deltaLength
            force[:,j] = temperature[:,j]*meridionalVelocity[:,j]/relativeVelocity[:,j]*deltaEntropy/deltaLength
        self.meridionalFields['EntropyDerivative'] = entropyMeridionalDerivative
        
        # # profiles implementation
        # for j in range(force.shape[1]):
        #     entropyIn = np.interp(normalizedSpan[0,j], self.spanwiseProfiles[0]['NormalizedSpan'], self.spanwiseProfiles[0]['Entropy'])
        #     entropyOut = np.interp(normalizedSpan[-1,j], self.spanwiseProfiles[1]['NormalizedSpan'], self.spanwiseProfiles[1]['Entropy'])
            
        #     # length of the streamline according to where data was extracted for the spanwise profiles
        #     streamCoordIn = np.interp(normalizedSpan[0,j], self.spanwiseProfiles[0]['NormalizedSpan'], self.spanwiseProfiles[0]['Streamwise_Coordinate'])
        #     streamCoordOut = np.interp(normalizedSpan[-1,j], self.spanwiseProfiles[1]['NormalizedSpan'], self.spanwiseProfiles[1]['Streamwise_Coordinate'])
        #     streamLength = streamCoordOut-streamCoordIn
            
        #     # length according to the blade only, as if the previous change should be applied only on the blade
        #     streamLength2 = streamCoords[-1,j]-streamCoords[0,j]
            
        #     # compute the entropy derivative and the force
        #     entropyMeridionalDerivative[:,j] = (entropyOut-entropyIn)/streamLength2
        #     force[:,j] = temperature[:,j]*meridionalVelocity[:,j]/relativeVelocity[:,j]*entropyMeridionalDerivative[:,j]
        
        self.meridionalFields['EntropyDerivative'] = entropyMeridionalDerivative
        
        return force
    
    
    def ComputeTangentialForceMarble(self, method='distributed'):
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
            drut_dz, drut_dr = compute_gradient_least_square(zgrid, rgrid, rgrid*tangentialVelocity)
            force = (drut_dz*self.meridionalFields['Velocity_Axial']+drut_dr*self.meridionalFields['Velocity_Radial'])/rgrid
        
        elif method=='distributed':
            force = np.zeros_like(meridionalVelocity)
            for j in range(meridionalVelocity.shape[1]):
                deltaForce = rgrid[-offset,j]*tangentialVelocity[-offset,j] - rgrid[offset,j]*tangentialVelocity[offset,j]
                deltaLength = streamCoords[-offset,j]-streamCoords[offset,j]
                force[:,j] = meridionalVelocity[:,j]/rgrid[:,j]*deltaForce/deltaLength
        else:
            raise ValueError('Method unknown')
        
        self.meridionalFields['AngularMomentumDerivative'] = force*rgrid/meridionalVelocity
        
        # second implementation based on profiles
        # self.meridionalFields['AngularMomentumDerivative'] = np.zeros_like(self.meridionalFields['Axial_Coordinate'])
        # normalizedSpan = compute_meridional_spanwise_coordinates(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], normalize=True)
        # force = np.zeros_like(meridionalVelocity)
        # for j in range(meridionalVelocity.shape[1]):
        #     angularMomentumIn = np.interp(normalizedSpan[0,j], self.spanwiseProfiles[0]['NormalizedSpan'], self.spanwiseProfiles[0]['AngularMomentum'])
        #     angularMomentumOut = np.interp(normalizedSpan[-1,j], self.spanwiseProfiles[1]['NormalizedSpan'], self.spanwiseProfiles[1]['AngularMomentum'])
        #     deltaMomentum = angularMomentumOut-angularMomentumIn
            
        #     # length according the extraction spanwise profiles
        #     streamCoordIn = np.interp(normalizedSpan[0,j], self.spanwiseProfiles[0]['NormalizedSpan'], self.spanwiseProfiles[0]['Streamwise_Coordinate'])
        #     streamCoordOut = np.interp(normalizedSpan[-1,j], self.spanwiseProfiles[1]['NormalizedSpan'], self.spanwiseProfiles[1]['Streamwise_Coordinate'])
        #     deltaLength = streamCoordOut-streamCoordIn
            
        #     # length according to the blade only grid
        #     deltaLength2 = streamCoords[-1,j]-streamCoords[0,j]
            
        #     # compute the force and the angular momentum derivative
        #     force[:,j] = meridionalVelocity[:,j]/rgrid[:,j]*deltaMomentum/deltaLength2
        #     self.meridionalFields['AngularMomentumDerivative'][:,j] = deltaMomentum/deltaLength
                
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
    
    
    def ComputeCalibrationCoefficients(self, calibration_method, metal_angle):
        if calibration_method.lower()=='lift/drag':
            self.ComputeLiftDragCalibrationCoefficients(metal_angle)
    
    
    def ComputeLiftDragCalibrationCoefficients(self, metal_angle):
        nBlades = self.config.get_blades_number()[self.iblade]
        circumferentialPitch = 2 * np.pi * self.radialGrid / nBlades
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