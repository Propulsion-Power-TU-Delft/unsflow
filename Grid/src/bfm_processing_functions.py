import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, sqrt, arctan2, arcsin, arccos, pi
import pickle
from grid.src.functions import *
import os
from scipy.interpolate import interp1d
import pandas as pd


# SOME GLOBAL QUANTITIES
gamma = 1.4
Rgas = 287
cp = (gamma*Rgas)/(gamma-1)
pRef = 101325
TRef = 288.15
ptIn = pRef
TtIn = 288.15
htIn = cp * TtIn
sIn = cp * np.log(TtIn / TRef) - Rgas * np.log(ptIn / pRef)






def perform_chima_profiles_processing(inputFiles, massFlows, refFile, visualDebug = True, weightedAveraging = True):
    """Perform the processing of Chima spanwise profiles, needed to inform Chima BF model.
    For the moment it assumes a single blade row, with uniform and standard inlet conditions.

    Args:
        inputFiles (list of strings): paths to pickle files containing the Chima spanwise profiles extracted from CFD
        
        massFlows (list of floats): mass flow values corresponding to each input file
        
        refFile (string): number of the reference case to be used for non-dimensionalization
    """
    # open the pickle file selected
    datas = []
    for inputFile in inputFiles:
        with open(inputFile, 'rb') as f:
            singleData = pickle.load(f)    
        datas.append(singleData)

    nOperatingPoints = len(datas)
    
    if visualDebug:
        for key in datas[0].keys():
            plt.figure()
            for ii, data in enumerate(datas):
                plt.plot(data[key], data['Span'], label = r'$\dot{m}/\dot{m}_{ref}=%.2f$' %(massFlows[ii]/massFlows[refFile]))
            plt.xlabel(key)
            plt.ylabel('Span')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.grid(alpha=0.2)
    
    
    # compute Delta quantities for all cases
    for ii in range(len(datas)):
        data = datas[ii]
        # data['Dht_Dm'] = (data['DeltaAngularMomentum']*omega)/data['Stream_Length']
        data['Dht_Dm'] = (data['Tot_Enthalpy'] - cp*TtIn)/data['Stream_Length']
        data['Ds_Dm'] = (data['Entropy'] - sIn)/data['Stream_Length']
        data['Deviation_Out'] = data['Deviation_Angle']
    
    
    # Compute the scaling functions with respect to the reference operating point
    phiTurning = np.zeros(nOperatingPoints)
    phiDeviation = np.zeros(nOperatingPoints)
    phiLoss = np.zeros(nOperatingPoints)
    massFlowNorm = np.zeros(nOperatingPoints)

    # aliases for reference quantities
    dht_dm_ref = datas[refFile]['Dht_Dm']
    ds_dm_ref = datas[refFile]['Ds_Dm']
    deviation_out_ref = datas[refFile]['Deviation_Out']
    spanwise_massflow_ref = datas[refFile]['Spanwise_Mass_Flow']
    if weightedAveraging != True:
        massflow_ref = np.ones_like(spanwise_massflow_ref)

    def weighted_average(values, weights):
        if weightedAveraging != True:
            weights = np.ones_like(values)
        avg = np.sum(values * weights) / np.sum(weights)
        return avg

    for i in range(len(datas)):
        massFlowNorm[i] = massFlows[i] / massFlows[refFile]
        
        phiTurning[i] = (weighted_average(datas[i]['Dht_Dm'], datas[i]['Spanwise_Mass_Flow']) /   
                         weighted_average(datas[refFile]['Dht_Dm'], datas[refFile]['Spanwise_Mass_Flow']))
        
        phiLoss[i] = (weighted_average(datas[i]['Ds_Dm'], datas[i]['Spanwise_Mass_Flow']) / 
                      weighted_average(datas[refFile]['Ds_Dm'], datas[refFile]['Spanwise_Mass_Flow']))
        
        phiDeviation[i] = (weighted_average(datas[i]['Deviation_Out'], datas[i]['Spanwise_Mass_Flow']) / 
                           weighted_average(datas[refFile]['Deviation_Out'], datas[refFile]['Spanwise_Mass_Flow']))

    # convert to numpy arrays
    massFlowNorm = np.array(massFlowNorm)
    phiTurning = np.array(phiTurning)
    phiLoss = np.array(phiLoss)
    phiDeviation = np.array(phiDeviation)
        
    if visualDebug:
        # scaling function for the turning model (turning type)
        plt.figure()
        plt.plot(massFlowNorm, phiTurning, '-o')
        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm turning}$ [-]')

        # scaling function for the loss model
        plt.figure()
        plt.plot(massFlowNorm, phiLoss, '-o')
        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm loss}$ [-]')

        # scaling function for the turning model (deviation type)
        plt.figure()
        plt.plot(massFlowNorm, phiDeviation, '-o')
        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm deviation}$ [-]')
    
    
    # now extrapolated the line data by 20% through linear extrapolation
    extrapExtent = 0.2
    deltaMass = massFlowNorm.max() - massFlowNorm.min()
    mMin = massFlowNorm.min() - deltaMass*extrapExtent
    mMax = massFlowNorm.max() + deltaMass*extrapExtent
    mFlowNew = np.linspace(mMin, mMax, 250)
    phiTurnNew = np.interp(mFlowNew, massFlowNorm, phiTurning)
    phiLossNew = np.interp(mFlowNew, massFlowNorm, phiLoss)

    # Create interpolation functions with extrapolation enabled
    phi_turning_fn = interp1d(massFlowNorm, phiTurning, kind='linear', fill_value='extrapolate')
    phi_loss_fn = interp1d(massFlowNorm, phiLoss, kind='linear', fill_value='extrapolate')
    phi_deviation_fn = interp1d(massFlowNorm, phiDeviation, kind='linear', fill_value='extrapolate')

    # Evaluate interpolated (and extrapolated) values
    phi_turning_new = phi_turning_fn(mFlowNew)
    phi_loss_new = phi_loss_fn(mFlowNew)
    phi_deviation_new = phi_deviation_fn(mFlowNew)

    if visualDebug:
        plt.figure()
        plt.plot(mFlowNew, phi_turning_new, '-')
        plt.axvline(x=massFlowNorm.min(), color='red', linestyle='--', linewidth=0.5)
        plt.axvline(x=massFlowNorm.max(), color='red', linestyle='--', linewidth=0.5)
        plt.scatter(1,1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm turning}$ [-]')

        plt.figure()
        plt.plot(mFlowNew, phi_loss_new, '-')
        plt.axvline(x=massFlowNorm.min(), color='red', linestyle='--', linewidth=0.5)
        plt.axvline(x=massFlowNorm.max(), color='red', linestyle='--', linewidth=0.5)
        plt.scatter(1,1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm loss}$ [-]')

        plt.figure()
        plt.plot(mFlowNew, phi_deviation_new, '-')
        plt.axvline(x=massFlowNorm.min(), color='red', linestyle='--', linewidth=0.5)
        plt.axvline(x=massFlowNorm.max(), color='red', linestyle='--', linewidth=0.5)
        plt.scatter(1,1)
        plt.grid(alpha=0.2)
        plt.xlabel(r'$\dot{m}/\dot{m}_{\rm ref}$ [-]')
        plt.ylabel(r'$\phi_{\rm deviation}$ [-]')
    
    
    
    # output the files nedeed by CTurboBFM
    profiles = {}
    profiles['Dht_Dm'] = datas[refFile]['Dht_Dm']
    profiles['Ds_Dm'] = datas[refFile]['Ds_Dm']
    profiles['Deviation_Out'] = datas[refFile]['Deviation_Out']
    profiles['Span_Normalized'] = datas[refFile]['Span']
    
    os.makedirs('Output', exist_ok=True)
    
    # output the profiles
    with open('Output/profiles_at_reference.pkl', 'wb') as handle:
        pickle.dump(profiles, handle)
    
    # output the scaling functions
    with open('Output/scaling_functions.csv', 'w') as f:
        f.write(f"MassFlow,PhiTurn,PhiLoss,PhiDev\n")
        for i in range(len(mFlowNew)):
            f.write(f"{mFlowNew[i]*massFlows[refFile]},{phi_turning_new[i]},{phi_loss_new[i]},{phi_deviation_new[i]}\n")
    print("Chima profiles processing completed. Files saved in 'Output' folder.")
    
    if visualDebug:
        plt.show()







def extract_chima_profiles_from_CFD(bladePklPath, inputAvgFile, visualDebug = True):
    """Extract the spanwise profiles needed for the Chima model

    Args:
        bladePklPath (str): path to the pickle multiblock file containing the blade geometry. The blade is assumed to be the first one in the multiblock file.
        inputAvgFile (str): path to file containting the circumferentially averaged CFD results. The outlet station is assumed to be the last one in the file.
        visualDebug (bool, optional): 

    Returns:
        None: save the profile pkl in the current folder
    """
        
    # read CFD avg file
    with open(inputAvgFile, 'rb') as f:
        data = pickle.load(f)

    # read blade file
    with open(bladePklPath, 'rb') as f:
        mb = pickle.load(f)
    blade = mb.blades[0]

    # compute additional fields
    data['Span'] = compute_meridional_spanwise_coordinates(
        data['Axial_Coordinate'], 
        data['Radial_Coordinate'], 
        normalize=True)
    data['Total_Pressure'] = data['Pressure (Pa)'] * (1 + (gamma - 1)/2*data['Mach']**2) ** (gamma / (gamma - 1))
    data['Beta_tt'] = data['Total_Pressure'] / ptIn
    data['Total_Temperature'] = data['Temperature (K)'] * (1 + (gamma - 1)/2*data['Mach']**2) 
    data['Total_Enthalpy'] = data['Total_Temperature'] * cp
    data['Tau_tt'] = data['Total_Temperature'] / TtIn
    data['Efficiency'] = (data['Beta_tt']**((gamma-1)/gamma)-1) / (data['Tau_tt']-1)
    data['Entropy'] = cp * np.log(data['Total_Temperature'] / TRef) - Rgas * np.log(data['Total_Pressure'] / pRef)
    data['Velocity_Meridional'] = np.sqrt(data['Velocity_Axial']**2 + data['Velocity_Radial']**2)
    data['Flow_Angle'] = np.arctan2(data['Velocity_Tangential_Relative'], data['Velocity_Meridional'])
    data['Angular_Momentum'] = data['Velocity_Tangential'] * data['Radial_Coordinate']


    def ComputeNormalMomentum(Density, Velocity_Axial, Velocity_Radial, Axial_Coordinate, Radial_Coordinate):
        ni, nj = Density.shape
        momentum = np.zeros_like(Density)
        for istream in range(ni):
        
            daxial = np.zeros_like(Axial_Coordinate[istream,:])
            dradial = np.zeros_like(Radial_Coordinate[istream,:])
            
            daxial[1:-1] = 0.5 * (Axial_Coordinate[istream,2:] - Axial_Coordinate[istream,0:-2])
            daxial[0] = Axial_Coordinate[istream,1] - Axial_Coordinate[istream,0]
            daxial[-1] = Axial_Coordinate[istream,-1] - Axial_Coordinate[istream,-2]
            
            dradial[1:-1] = 0.5 * (Radial_Coordinate[istream,2:] - Radial_Coordinate[istream,0:-2])
            dradial[0] = Radial_Coordinate[istream,1] - Radial_Coordinate[istream,0]
            dradial[-1] = Radial_Coordinate[istream,-1] - Radial_Coordinate[istream,-2]
            
            dthetaExtension = 2 * np.pi * Radial_Coordinate[istream,:]
            
            
            for ispan in range(nj):
                surfaceVector = np.array([dradial[ispan], -daxial[ispan]])*dthetaExtension[ispan]
                velocity = np.array([Velocity_Axial[istream, ispan], Velocity_Radial[istream, ispan]])
                momentum[istream, ispan] = Density[istream, ispan] * np.dot(velocity, surfaceVector)
        return momentum

    data['Spanwise_Mass_Flow'] = ComputeNormalMomentum(data['Density (kg/m³)'], data['Velocity_Axial'], data['Velocity_Radial'], data['Axial_Coordinate'], data['Radial_Coordinate'])


    if visualDebug:
        plt.figure()
        plt.plot(data["Beta_tt"][-1,:], data['Span'][-1,:], '--o', mfc='none', label='Current')
        plt.xlabel(r'$\beta_{\rm tt} \ \rm{[-]}$')
        plt.ylabel(r'Span [-]')
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data["Tau_tt"][-1,:], data['Span'][-1,:], '--o', mfc='none', label='Current')
        plt.xlabel(r'$\tau_{\rm tt} \ \rm{[-]}$')
        plt.ylabel(r'Span [-]')
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(data["Efficiency"][-1,:], data['Span'][-1,:], '--o', mfc='none', label='Current')
        plt.xlabel(r'$\eta_{\rm tt} \ \rm{[-]}$')
        plt.ylabel(r'Span [-]')
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()

        # analysis of trailing edge
        plt.figure()
        plt.plot(blade.blade_metal_angle[-1,:]*180/pi, blade.spanwise_normalized_coord[-1,:], label=r'$\kappa_{\rm te}$')
        plt.plot(data['Flow_Angle'][-1,:]*180/pi, data['Span'][-1,:], '--o', mfc='none', label=r'$\beta_{\rm te}$')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.ylabel('Span [-]')
        plt.xlabel('Angles [deg]')
        plt.tight_layout()


    # now export what is needed by the Chima model. 
    chima = {}
    chima['Span'] = data['Span'][-1,:]
    chima['Radius'] = data['Radial_Coordinate'][-1,:]
    chima['Stream_Length'] = blade.streamwise_coord[-1,:] 
    chima['Vel_Tang'] = data['Velocity_Tangential'][-1,:]
    chima['Entropy'] = data['Entropy'][-1,:]
    chima['Tot_Enthalpy'] = data['Total_Enthalpy'][-1,:]
    chima['Deviation_Angle'] = data['Flow_Angle'][-1,:] - blade.blade_metal_angle[-1,:]
    chima['DeltaAngularMomentum'] = data['Angular_Momentum'][-1,:]
    chima['Spanwise_Mass_Flow'] = data['Spanwise_Mass_Flow'][-1,:]
    chima['Flow_Angle'] = data['Flow_Angle'][-1,:]
    chima['Vel_Meridional'] = data['Velocity_Meridional'][-1,:]

    with open("exit_profile.pkl", 'wb') as f:
        pickle.dump(chima, f)
        print("Exit profile saved to exit_profile.pkl")
    
    if visualDebug:
        plt.show()



def read_cturbobfm_csv_file(gridFilePath):
    """Return the pandas dataframe of a cturbobfm csv file

    Args:
        gridFilePath (str): path to the grid csv file

    Returns:
        pandas.DataFrame: dataframe containing the grid data
    """
    # Read the first three lines to extract grid sizes
    with open(gridFilePath, 'r') as f:
        ni = int(f.readline().strip().split('=')[1])
        nj = int(f.readline().strip().split('=')[1])
        nk = int(f.readline().strip().split('=')[1])

    # Read the rest of the CSV data into a DataFrame
    df = pd.read_csv(gridFilePath, skiprows=3)

    return df, ni, nj, nk





def merge_chima_profiles_with_cturbobfm_grid(gridFilePath, profilesPath, visualDebug = True):
    """Merge the Chima profile (at reference) onto the cturbobfm grid to write a complete grid file to be run with the Chima model

    Args:
        gridFilePath (str): path to the cturbobfm grid file
        profilesPath (str): path to the pickle file containing the Chima profiles at reference
        visualDebug (bool, optional): 
    """
    
    # read grid
    df, ni, nj, nk = read_cturbobfm_csv_file(gridFilePath)
    
    # read profiles
    with open(profilesPath, 'rb') as f:
        profiles = pickle.load(f)
    
    span = df['spanwiseLength'].to_numpy()
    span = np.reshape(span, (ni, nj, nk))

    bladePresent = df['bladePresent'].to_numpy()
    bladePresent = np.reshape(bladePresent, (ni, nj, nk))

    numberBlades = df['numberBlades'].to_numpy()
    numberBlades = np.reshape(numberBlades, (ni, nj, nk))
    
    # interpolate profiles values onto the grid
    dht_dm = np.zeros((ni, nj, nk))
    ds_dm = np.zeros((ni, nj, nk))
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                if numberBlades[i,j,k] > 0:
                    dht_dm[i,j,k] = np.interp(span[i,j,k], profiles['Span_Normalized'], profiles['Dht_Dm'])
                    ds_dm[i,j,k] = np.interp(span[i,j,k], profiles['Span_Normalized'], profiles['Ds_Dm'])
    
    
    
    if visualDebug:
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        z = df['z'].to_numpy()
        x = np.reshape(x, (ni, nj, nk))
        y = np.reshape(y, (ni, nj, nk))
        
        plt.figure()
        plt.contourf(x[:,:,0], y[:,:,0], dht_dm[:,:,0], levels = 50, cmap='jet')
        plt.gca().set_aspect('equal')
        plt.title('Dht_Dm [J/kgm]')
        plt.colorbar()
        
        plt.figure()
        plt.contourf(x[:,:,0], y[:,:,0], ds_dm[:,:,0], levels = 50, cmap='jet')
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.title('Ds_Dm [J/kgKm]')
    
    
    # now add this new 3D arrays to the grid dataframe and save the new grid file
    df['Dht_Dm'] = dht_dm.flatten()
    df['Ds_Dm'] = ds_dm.flatten()
    
    output_file = gridFilePath.replace('.csv', '_withChimaProfiles.csv')

    with open(output_file, 'w') as f:
        f.write(f"NI={ni}\n")
        f.write(f"NJ={nj}\n")
        f.write(f"NK={nk}\n")        
        df.to_csv(f, index=False)
    print(f"New grid file with Chima profiles saved to {output_file}")
        