import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI, PhaseSI
from unsflow.utils.thesis_plots import *


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
    ax.fill_betweenx(sat_curve['T_sat'], sat_curve['s_liq'], sat_curve['s_vap'], color='lightgray', alpha=1.0)


def plot_isobars(ax, ax2, isobars):
    T_isobars = isobars['T']
    s_isobars = isobars['s']
    for p in s_isobars.keys():
        ax.plot(s_isobars[p], T_isobars, '--C1', lw=0.6)
        ax2.axvline(p/1e5, color='C1', linestyle='--', lw=0.6)
        # ax.text(s_isobars[p][-1], T_isobars[-1]+1, f'{p/1E5:.0f}', color='C1')


def compute_isobars(T_min, T_max, P_isobars, fluid, N):
    T_isobars = np.linspace(T_min, T_max, N)
    s_isobars = {}
    for p in P_isobars:
        s_isobars[p] = PropsSI('S', 'T', T_isobars, 'P', p, fluid)
    return {'T': T_isobars, 's': s_isobars}


def compute_quantity_s_p(quantity, s, p, fluid):
    if quantity == 'gamma_pv':
        rho = PropsSI('D', 'S', s, 'P', p, fluid)
        dp_drho_s = PropsSI('d(P)/d(Dmass)|Smass','S', s, 'P', p, fluid)
        gamma_pv = rho / p * dp_drho_s
        q = gamma_pv
    elif quantity == "fundamental_derivative":
        q = PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'P', p, 'S', s, fluid)
    else:
        q = PropsSI(quantity, 'S', s, 'P', p, fluid)
    return q

def compute_quantity_p_T(quantity, p, T, fluid):
    if quantity == 'gamma_pv':
        rho = PropsSI('D', 'P', p, 'T', T, fluid)
        dp_drho_s = PropsSI('d(P)/d(Dmass)|Smass','P', p, 'T', T, fluid)
        gamma_pv = rho / p * dp_drho_s
        q = gamma_pv
    elif quantity == "fundamental_derivative":
        q = PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'P', p, 'T', T, fluid)
    else:
        q = PropsSI(quantity, 'P', p, 'T', T, fluid)
    return q

    
def compute_two_dimensional_values_Ts(P_isobars, sat_curve, quantity, fluid, T_min, T_max, N):
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
        s_grid[i, :] = np.linspace(s_min, s_max, N)
        qt_grid[i, :] = compute_quantity_s_p(quantity, s_grid[i, :], p, fluid)
        T_grid[i, :] = PropsSI('T', 'S', s_grid[i, :], 'P', p, fluid)

    return {'T': T_grid, 's': s_grid, 'qt': qt_grid}


def compute_two_dimensional_values_PT(P_isobars, quantity, fluid, T_min, T_max, N):
    P_range = np.linspace(P_isobars[0], P_isobars[-1], N)   
    T_range = np.linspace(T_min, T_max, N)   
    qt_grid = np.zeros((N, N))
    P_grid, T_grid = np.meshgrid(P_range, T_range, indexing='ij')

    # first axis is pressure
    for i in range(N):
        p = P_range[i]
        pc = PropsSI('Pcrit', fluid)
        if p<pc:
            T_sat = PropsSI('T', 'P', p, 'Q', 1, fluid)
            if T_min<T_sat:
                T_grid[i, :] = np.linspace(T_sat+0.001, T_max, N)
        else:
            pass
        qt_grid[i, :] = compute_quantity_p_T(quantity, p, T_grid[i, :], fluid)

    return {'P': P_grid, 'T': T_grid, 'qt': qt_grid}


def get_math_label(name):
    name = name.lower()
    dictnames = {
        'a': r'$a~[\rm m/s]$',
        'p': r'$p~[\rm Pa]$',
        'h': r'$h~[\rm J/kgK]$',
        'gamma_pv': r'$\gamma_{pv}$',
        'd': r'$\rho~[\rm kg/m^3]$',
        'z': r'$Z$',
        'fundamental_derivative': r'$\Gamma$',
        's': r'$s~[\rm J/kgK]$',
        'T': r'$T~[\rm K]$'
    }

    if name in dictnames.keys():
        return dictnames[name]
    else:
        raise ValueError(f"Quantity {name} unknown")
    

def plot_thermo_diagrams(fluid, T_min, T_max, P_isobars, quantity, N=50):

    set_thesis_style()
    fig, axs = create_figure(1, 1.3, (1, 2), sharey=True)
    axs[0].set_xlabel(r'$s$ [J/kgK]')
    axs[0].set_ylabel(r'$T$ [K]')
    axs[1].set_xlabel(r'$p$ [bar]')
    # axs[1].set_ylabel(r'$T$ [K]')
    label = get_math_label(quantity)

    # compute critical values
    Tc = PropsSI('Tcrit', fluid)
    Pc = PropsSI('Pcrit', fluid)
    print(f"Information for fluid: {fluid}")
    print(f"Critical pressure: {Pc/1e5:.2f} bar, and temperature: {Tc:.1f} K")
    
    # compute saturation curve
    sat_curve = compute_saturation_curve_TS(fluid, T_min)
    

    # compute isobar curves
    isobars = compute_isobars(T_min, T_max, P_isobars, fluid, 10*N)
    plot_isobars(axs[0], axs[1], isobars)

    TS_values = compute_two_dimensional_values_Ts(P_isobars, sat_curve, quantity, fluid, T_min, T_max, N)
    cnt0 = axs[0].contourf(TS_values['s'], TS_values['T'], TS_values['qt'], levels=30)
    # cbar0 = plt.colorbar(cnt0)
    # cbar0.set_label(label)

    PT_values = compute_two_dimensional_values_PT(P_isobars, quantity, fluid, T_min, T_max, N)
    cnt1 = axs[1].contourf(PT_values['P']/1e5, PT_values['T'], PT_values['qt'], levels=30)
    cbar = fig.colorbar(cnt0, ax=axs, label=label)
    cbar.set_label(label)

    plot_saturation_dome(axs[0], sat_curve)

    return fig, axs

def include_polytropic_transformation(fig, axs, p1, T1, p2, T2, fluid, c='r'):
    k = np.log(p2/p1)/np.log(T2/T1)
    n = k/(k-1)
    T = np.linspace(T1, T2, 100)
    p = p1 * (T/T1)**(n/(n-1))
    s = np.zeros_like(p)
    for i in range(len(p)):
        s[i] = PropsSI('S', 'P', p[i], 'T', T[i], fluid)

    ms = 2
    axs[0].plot(s, T, '-%s' %c)
    axs[0].plot(s[0], T[0], 'o%s' %c, ms=ms)
    axs[0].plot(s[-1], T[-1], 'x%s' %c, ms=ms)

    axs[1].plot(p/1e5, T, '-%s' %c)
    axs[1].plot(p[0]/1e5, T[0], 'o%s' %c, ms=ms)
    axs[1].plot(p[-1]/1e5, T[-1], 'x%s' %c, ms=ms)
    



fig, axs = plot_thermo_diagrams('R1233zd(E)', T_min=280, T_max=620, P_isobars=[1e5, 5e5, 15e5, 25e5, 30e5], quantity='gamma_pv', N=100)
p1, T1 = 5e5, 355
p2, T2 = 20.04e5, 433
include_polytropic_transformation(fig, axs, p1, T1, p2, T2, 'R1233zd(E)', 'r')

# plot_thermo_diagrams('MM', T_min=350, T_max=620, P_isobars=[1e5, 5e5, 15e5, 18e5], quantity='gamma_pv', N=100)
# plot_thermo_diagrams('CH4', T_min=100, T_max=250, P_isobars=[1e5, 5e5, 15e5, 40e5], quantity='gamma_pv', N=100)
plt.show()