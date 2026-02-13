import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI


def compute_work_coefficient_temperature_ratio(tratio, r_ref, omega=None, rpm=None, cp=1005, Tt1=288.15):
    if (omega is None and rpm is None):
        raise ValueError("either omega or rpm must be provided")
    elif (rpm is not None):
        omega = 2 * np.pi * rpm / 60
    else:
        pass
    
    u_ref = omega * r_ref
    psi = cp*(tratio - 1)*Tt1 / (u_ref**2)
    return psi

def compute_flow_coefficient(mflow, density, r_ref, omega=None, rpm=None):
    if (omega is None and rpm is None):
        raise ValueError("either omega or rpm must be provided")
    elif (rpm is not None):
        omega = 2 * np.pi * rpm / 60
    else:
        pass
    
    d_ref = 2*r_ref
    u_ref = omega * r_ref
    phi = mflow / (density * d_ref**2 * u_ref)
    return phi

def compute_tip_mach_number(omega=None, rpm=None, r_ref=None, gamma=1.4, Tt1=288.15, Rgas=287):
    if (omega is None and rpm is None):
        raise ValueError("either omega or rpm must be provided")
    elif (rpm is not None):
        omega = 2 * np.pi * rpm / 60
    else:
        pass
    
    u_ref = omega * r_ref
    M_tip = u_ref / np.sqrt(gamma * Rgas * Tt1)
    return M_tip

def compute_isentropic_efficiency(pratio, tratio, gamma=1.4):
    eta = (pratio**((gamma-1)/gamma) - 1) / (tratio - 1)
    return eta

def compute_temperature_ratio(pratio, efficiency, gamma=1.4):
    tratio = 1 + (pratio**((gamma-1)/gamma) - 1) / efficiency
    return tratio

def compute_work_coefficient_temperature_ratio_real(delta_ht, r_ref, omega=None, rpm=None):
    if (omega is None and rpm is None):
        raise ValueError("either omega or rpm must be provided")
    elif (rpm is not None):
        omega = 2 * np.pi * rpm / 60
    else:
        pass
    
    u_ref = omega * r_ref
    psi = delta_ht / (u_ref**2)
    return psi

def compute_tip_mach_number_real(r_ref, soundspeed, omega=None, rpm=None):
    if (omega is None and rpm is None):
        raise ValueError("either omega or rpm must be provided")
    elif (rpm is not None):
        omega = 2 * np.pi * rpm / 60
    else:
        pass
    
    u_ref = omega * r_ref
    M_tip = u_ref / soundspeed
    return M_tip