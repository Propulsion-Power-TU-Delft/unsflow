from Grid.src.config import Config
import os
import numpy as np
from numpy import sin, cos, tan, arctan2, pi, sqrt
import pandas as pd
from Grid.src.functions import contour_template, compute_meridional_streamwise_coordinates, compute_gradient_least_square, compute_meridional_spanwise_coordinates

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
        self.meridionalFields = self.ProcessParaviewSU2Dataset(folder_path=datasetPath, average_type=averageType)
            
    
    def ProcessParaviewSU2Dataset(self, folder_path, average_type, CP=1005, R=287, TREF=288.15, PREF=101300):
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
        saveFields = ['Density', 'Energy', 'Mach', 'Eddy_Viscosity', 'Pressure', 'Temperature',
                      'Grid_Velocity_Tangential', 'Velocity_Radial', 'Velocity_Tangential',
                      'Velocity_Tangential_Relative', 'Velocity_Axial', 'Entropy']
        
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

            x = data_dict['Points_0']
            y = data_dict['Points_1']
            z = data_dict['Points_2']
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            stream_id, span_id = extract_grid_location(file)
            z_grid[stream_id, span_id] = np.sum(z) / len(z)
            r_grid[stream_id, span_id] = np.sum(r) / len(r)

            # Compute additional fields that will be circumferentially averaged 
            data_dict['Velocity_Radial'] = data_dict['Velocity_0']*cos(theta)+data_dict['Velocity_1']*sin(theta)

            data_dict['Velocity_Tangential'] = -data_dict['Velocity_0']*sin(theta)+data_dict['Velocity_1']*cos(theta)

            data_dict['Grid_Velocity_Tangential'] = -data_dict['Grid_Velocity_0']*sin(theta)+data_dict['Grid_Velocity_1']*cos(theta)

            data_dict['Velocity_Tangential_Relative'] = data_dict['Velocity_Tangential']-data_dict['Grid_Velocity_Tangential'] 
            
            data_dict['Velocity_Axial'] = data_dict['Velocity_2']

            data_dict['Entropy'] = CP*np.log(data_dict['Temperature']/TREF)-R*np.log(data_dict['Pressure']/PREF)

            for fieldName in saveFields:
                f = data_dict[fieldName].copy()
                if self.avg_type.lower() == 'raw':
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f) / len(f)
                elif self.avg_type.lower() == 'density':
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f * data_dict['Density']) / np.sum(data_dict['Density'])
                elif self.avg_type.lower() == 'axial_momentum':
                    meridionalFields[fieldName][stream_id, span_id] = np.sum(f * data_dict['Momentum_2']) / np.sum(data_dict['Momentum_2'])

        meridionalFields['Axial_Coordinate'] = z_grid
        meridionalFields['Radial_Coordinate'] = r_grid
        return meridionalFields
    
    
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

        self.bodyForceFields['Force_Axial'] = self.bodyForceFields['Force_Viscous_Axial']-(self.bodyForceFields['Force_Tangential']-
                                                self.bodyForceFields['Force_Viscous_Tangential'])*self.meridionalFields['Velocity_Tangential_Relative']/self.meridionalFields['Velocity_Axial']
        self.bodyForceFields['Force_Inviscid_Axial'] = self.bodyForceFields['Force_Axial']-self.bodyForceFields['Force_Viscous_Axial']

        # self.meridional_fields['Force_Inviscid'] = np.sqrt(self.meridional_fields['Force_Inviscid_Axial']**2+
        #                                                    self.meridional_fields['Force_Inviscid_Radial']**2+
        #                                                    self.meridional_fields['Force_Inviscid_Tangential']**2)
        
        # self.meridional_fields['Force_Radial'] = self.meridional_fields['Force_Viscous_Radial']+self.meridional_fields['Force_Inviscid_Radial']
    
    
    def ComputeLossForceMarble(self):
        """Compute the loss force component according to Marble method

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