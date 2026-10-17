import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI


def compute_saturation_curve_TS(fluid, Tmin, N=200):
    Tc = PropsSI('Tcrit', fluid)
    T_sat = np.linspace(Tmin, Tc, N)
    S_liq = []
    S_vap = []
    for Ti in T_sat:
        try:
            P = PropsSI('P','T', Ti, 'Q', 0, fluid)
            S_l = PropsSI('S','T',Ti,'Q', 0, fluid)
            S_v = PropsSI('S','T',Ti,'Q', 1, fluid)
            
            S_liq.append(S_l)
            S_vap.append(S_v)
        except:
            continue
    S_liq = np.array(S_liq)
    S_vap = np.array(S_vap)
    sol = {'T_sat': T_sat, 's_liq': S_liq, 's_vap': S_vap}
    return sol

def plot_saturation_dome(ax, sat_curve):
    ax.plot(sat_curve['s_liq'], sat_curve['T_sat'], '-k')
    ax.plot(sat_curve['s_vap'], sat_curve['T_sat'], '-k')
    plt.fill_betweenx(sat_curve['T_sat'], sat_curve['s_liq'], sat_curve['s_vap'], color='lightgray', alpha=1.0)

def plot_isobars(ax, isobars):
    T_isobars = isobars['T']
    s_isobars = isobars['s']
    for p in s_isobars.keys():
        ax.plot(s_isobars[p], T_isobars, '--C1', lw=0.6)
        plt.text(s_isobars[p][-1], T_isobars[-1]+1, f'{p/1E5:.0f}', color='C1')

def compute_isobars(T_min, T_max, P_isobars, fluid, N):
    T_isobars = np.linspace(T_min, T_max, N)
    s_isobars = {}
    for p in P_isobars:
        s_isobars[p] = PropsSI('S', 'T', T_isobars, 'P', p, fluid)
    return {'T': T_isobars, 's': s_isobars}
    
def compute_two_dimensional_values(P_isobars, sat_curve, quantity, fluid, T_min, T_max, N):
    s_vap = sat_curve['s_vap']
    T_vap = sat_curve['T_sat']

    p_range = np.linspace(P_isobars[0], P_isobars[-1], N)   
    qt_grid = np.zeros((N, N))
    s_grid = np.zeros((N, N))
    T_grid = np.zeros((N, N))

    # first axis is pressure, second axis is entropy
    for i in range(N):
        p = p_range[i]
        pc = PropsSI('Pcrit', fluid)
        if p<pc:
            s_sat = PropsSI('S', 'P', p, 'Q', 1, fluid)
            s_min = s_sat+0.001
        else:
            s_min = PropsSI('S', 'P', p, 'T', T_min, fluid)

        s_max = PropsSI('S', 'P', p, 'T', T_max, fluid)
        s_grid[i,:] = np.linspace(s_min, s_max, N)
        for j in range(N):
            qt_grid[i, j] = PropsSI(quantity, 'S', s_grid[i,j], 'P', p, fluid)
            T_grid[i, j] = PropsSI('T', 'S', s_grid[i,j], 'P', p, fluid)

    return {'T': T_grid, 's': s_grid, 'qt': qt_grid}
    

def plot_TS_diagram(fluid, T_min, T_max, P_isobars, quantity, N=50):
    # plot used throughout the code
    fig, ax = plt.subplots(1)
    ax.set_xlabel(r'$s$ [J/kgK]')
    ax.set_ylabel(r'$T$ [K]')

    # compute critical values
    Tc = PropsSI('Tcrit', fluid)
    Pc = PropsSI('Pcrit', fluid)

    # compute saturation curve
    sat_curve = compute_saturation_curve_TS(fluid, T_min)
    

    isobars = compute_isobars(T_min, T_max, P_isobars, fluid, 10*N)
    plot_isobars(ax, isobars)

    grid_values = compute_two_dimensional_values(P_isobars, sat_curve, quantity, fluid, T_min, T_max, N)
    
    cnt = ax.contourf(grid_values['s'], grid_values['T'], grid_values['qt'], levels=40)
    cbar = plt.colorbar(cnt)
    cbar.set_label(quantity)
    plot_saturation_dome(ax, sat_curve)


plot_TS_diagram('MM', T_min=300, T_max=600, P_isobars=[1e4, 1e5, 10e5], quantity='D')
plt.show()