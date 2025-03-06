from Grid.src.config import Config
import os
import numpy as np
from numpy import sin, cos, tan, arctan2, pi, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from Grid.src.functions import contour_template, compute_meridional_streamwise_coordinates, compute_gradient_least_square, compute_meridional_spanwise_coordinates
import pickle


class BodyForce:
    """
    class that stores the information regarding the blade topology.
    """

    def __init__(self, config):
        self.config = config
        self.bodyForceFields = {} # contaning the body force fields
        self.meridionalFields = {} # contaning all the meridional fields
    
    
    def CircumferentialAverage(self):
        averageType = self.config.get_circumferential_average_type()
        datasetPath = self.config.get_paraview_macro_dataset_folderpath()
        solverType = self.config.get_bladed_CFD_solver_type()
        self.meridionalFields = self.ProcessParaviewDataset(folder_path=datasetPath, solver_type=solverType, average_type=averageType)
            
    
    def ProcessParaviewDataset(self, folder_path, solver_type, average_type, CP=1005, R=287, TREF=288.15, PREF=101300):
        """
        Read the processed dataset stored in folder_path location obtained by the Paraview Macro, for The Marble Extraction procedure.
        Average type distinguish the type of average used, raw for standard circumferential.
        Inviscid=True sets the viscous stresses to zero, leading to inviscid force extraction.
        """
        available_avg_types = ['raw', 'density', 'axialMomentum']
        self.avg_type = average_type.lower()
        if self.avg_type not in available_avg_types:
            raise ValueError('Not valid average type')
        print('Weighted average type: %s' % self.avg_type)
        
        available_solver_names = ['su2', 'luminary']
        if solver_type.lower() not in available_solver_names:
            raise ValueError('Not valid solver type')
        print('Solver type: %s' % solver_type)

        variablesNames = self.get_CFD_variable_names(solver_type)
        
        def extract_grid_location(file_name):
            print('Elaborating Filename: ' + file_name)
            file_name = file_name.strip('spline_data_')
            file_name = file_name.strip('.csv')
            file_name = file_name.split('_')
            nz = int(file_name[0])
            nr = int(file_name[1])
            return nz, nr
       
        data_dir = folder_path
        files = [f for f in os.listdir(data_dir) if '.csv' in f]
        files = sorted(files)

        # give the name of the fields to average
        saveFields = ['Density', 'Mach', 'Pressure', 'Temperature',
                      'Grid_Velocity_Tangential', 'Velocity_Radial', 'Velocity_Tangential',
                      'Velocity_Tangential_Relative', 'Velocity_Axial', 'Entropy', 'Relative_Flow_Angle']
        
        ni, nj = extract_grid_location(files[-1])
        streamPoints = ni+1
        spanPoints = nj+1
        meridionalFields = {}
        for field_name in saveFields:
            meridionalFields[field_name] = np.zeros((streamPoints, spanPoints))
        z_grid = np.zeros((streamPoints, spanPoints))
        r_grid = np.zeros((streamPoints, spanPoints))

        for file in files:
            df = pd.read_csv(data_dir + '/' + file)
            data_dict = df.to_dict('list')
            data_dict = {key: np.array(value) for key, value in data_dict.items()}

            x = data_dict[variablesNames['X']]
            y = data_dict[variablesNames['Y']]
            z = data_dict[variablesNames['Z']]
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            stream_id, span_id = extract_grid_location(file)
            z_grid[stream_id, span_id] = np.sum(z) / len(z)
            r_grid[stream_id, span_id] = np.sum(r) / len(r)

            # Compute additional fields that will be circumferentially averaged 
            data_dict['Density'] = data_dict[variablesNames['Density']]
            
            data_dict['Mach'] = data_dict[variablesNames['Mach']]
            
            data_dict['Pressure'] = data_dict[variablesNames['Pressure']]
            
            data_dict['Temperature'] = data_dict[variablesNames['Temperature']]
            
            data_dict['Velocity_Radial'] = data_dict[variablesNames['Velocity_X']]*cos(theta)+data_dict[variablesNames['Velocity_Y']]*sin(theta)

            data_dict['Velocity_Tangential'] = -data_dict[variablesNames['Velocity_X']]*sin(theta)+data_dict[variablesNames['Velocity_Y']]*cos(theta)

            data_dict['Grid_Velocity_Tangential'] = -data_dict[variablesNames['Grid_Velocity_X']]*sin(theta)+data_dict[variablesNames['Grid_Velocity_Y']]*cos(theta)

            data_dict['Velocity_Tangential_Relative'] = data_dict['Velocity_Tangential']-data_dict['Grid_Velocity_Tangential'] 
            
            data_dict['Velocity_Axial'] = data_dict[variablesNames['Velocity_Z']]

            data_dict['Entropy'] = CP*np.log(data_dict[variablesNames['Temperature']]/TREF)-R*np.log(data_dict[variablesNames['Pressure']]/PREF)

            data_dict['Relative_Flow_Angle'] = np.arctan2(data_dict['Velocity_Tangential_Relative'], data_dict['Velocity_Axial'])
            
            for fieldName in saveFields:
                f = data_dict[fieldName].copy()
                if self.avg_type.lower() == 'raw':
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f) / len(f)
                elif self.avg_type.lower() == 'density':
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f * data_dict[variablesNames['Density']]) / np.sum(data_dict[variablesNames['Density']])
                elif self.avg_type.lower() == 'axial_momentum':
                    axialMomentum = data_dict['Velocity_Axial'] * data_dict[variablesNames['Density']]
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f * axialMomentum) / np.sum(axialMomentum)

        meridionalFields['Axial_Coordinate'] = z_grid
        meridionalFields['Radial_Coordinate'] = r_grid
        return meridionalFields
    
    
    def get_CFD_variable_names(self, solver_type):
        """Return the dictionary with all the correct variable names depending on the solver employed for the CFD

        Args:
            solver_type (string): Luminary or SU2

        Returns:
            dict: dictionary of names
        """
        names = {}

        if solver_type.lower() == 'su2':
            names['X'] = 'Points_0'
            names['Y'] = 'Points_1'
            names['Z'] = 'Points_2'
            names['Density'] = 'Density'
            names['Velocity_X'] = 'Velocity_0'
            names['Velocity_Y'] = 'Velocity_1'
            names['Velocity_Z'] = 'Velocity_2'
            names['Pressure'] = 'Pressure'
            names['Temperature'] = 'Temperature'
            names['Grid_Velocity_X'] = 'Grid_Velocity_0'
            names['Grid_Velocity_Y'] = 'Grid_Velocity_1'
            names['Grid_Velocity_Z'] = 'Grid_Velocity_2'
            names['Mach'] = 'Mach'
        
        elif solver_type.lower() == 'luminary':
            names['X'] = 'Points_0'
            names['Y'] = 'Points_1'
            names['Z'] = 'Points_2'
            names['Density'] = 'Density (kg/m³)'
            names['Velocity_X'] = 'Velocity (m/s)_0'
            names['Velocity_Y'] = 'Velocity (m/s)_1'
            names['Velocity_Z'] = 'Velocity (m/s)_2'
            names['Pressure'] = 'Absolute Pressure (Pa)'
            names['Temperature'] = 'Temperature (K)'
            names['Grid_Velocity_X'] = 'Grid Velocity (m/s)_0'
            names['Grid_Velocity_Y'] = 'Grid Velocity (m/s)_1'
            names['Grid_Velocity_Z'] = 'Grid Velocity (m/s)_2'
            names['Mach'] = 'Mach'
        
        return names
    
    
    def PlotMeridionalFields(self):
        for name, field in self.meridionalFields.items():
            if name != 'Axial_Coordinate' and name!= 'Radial_Coordinate':
                contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, name)
    
    def PlotBodyForceFields(self):
        for name, field in self.bodyForceFields.items():
            contour_template(self.meridionalFields['Axial_Coordinate'], self.meridionalFields['Radial_Coordinate'], field, name)
    
    
    def ComputeBodyForceMarble(self):
        """
        Compute the body force density, using the marble thermodynamic approach based on the circumferentially averaged flow field
        """
        # ni,nj = self.meridionalFields['Axial_Coordinate'].shape
        
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
        # self.bodyForceFields['Force_Inviscid_Radial'] = np.abs(self.meridional_fields['Force_Inviscid_Tangential'])*np.tan(self.lean_angle)
        self.bodyForceFields['Force_Inviscid_Radial'] = np.abs(self.bodyForceFields['Force_Inviscid_Tangential'])*0

        self.bodyForceFields['Force_Axial'] = self.bodyForceFields['Force_Viscous_Axial']-(self.bodyForceFields['Force_Tangential']-
                                                self.bodyForceFields['Force_Viscous_Tangential'])*self.meridionalFields['Velocity_Tangential_Relative']/self.meridionalFields['Velocity_Axial']
        
        self.bodyForceFields['Force_Inviscid_Axial'] = self.bodyForceFields['Force_Axial']-self.bodyForceFields['Force_Viscous_Axial']

        self.bodyForceFields['Force_Inviscid'] = np.sqrt(self.bodyForceFields['Force_Inviscid_Axial']**2+
                                                           self.bodyForceFields['Force_Inviscid_Radial']**2+
                                                           self.bodyForceFields['Force_Inviscid_Tangential']**2)
        
        self.bodyForceFields['Force_Viscous'] = np.sqrt(self.bodyForceFields['Force_Viscous_Axial']**2+
                                                        self.bodyForceFields['Force_Viscous_Radial']**2+
                                                        self.bodyForceFields['Force_Viscous_Tangential']**2)
        # self.meridional_fields['Force_Radial'] = self.meridional_fields['Force_Viscous_Radial']+self.meridional_fields['Force_Inviscid_Radial']
        
        
    
    
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
        