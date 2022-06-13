import numpy as np
from scipy.special import binom
import pandas as pd


"Model parameter getter and setter"
def cordova_vgm_params(model_param_df, grid_xy, top_layer_depth, aux_data=None):
    """
    Prepare model parameters for Cordova model with van Genuchten-Mualem model
    Use a given CSV file that includes soil properties for each atomic element
    model representation in space (e.g., grid_xy)
    Parameters
    ----------
    model_param_df : Pandas Dataframe
        Soil properties for vGM model
    grid_xy : int
        Spatial Element of interest "int
    top_layer_depth: int
        Depth of top layer soil in millimeters 
    output : list
        List of Model parameters
        0 theta_s: Soil moisture at Saturation [mm]
        1 theta_wp: Soil moisture at Wilting Point [mm]
        2 theta_star: Soil moisture at point with ET regime change [mm]
        3 k_s: Hydraulic conductivity at saturation [mm/d]
        4 psi_s: Saturated soil matric potential [mm]
        5 vgm_n: VGM fit parameter
        6 porosity: Porosity [%]
        7 d_i: Diffusivity index
        8 ps_i: Pore Size index
        9 pc_i: Pore connectivity index
    """
    # find the index of the element in the dataframe
    idx = model_param_df['grid_xy'] == int(grid_xy)

    # Constant model parameters
    d_i = 5           # Diffusivity index
    ps_i = 0.286      # Pore Size index
    pc_i = 10         # pore connectivity index
    # print(model_param_df['k_s'][idx])
    # Spatially distributed paramaters
    k_s = model_param_df['k_s'][idx].values[0]
    
    psi_s = model_param_df['psi_s'][idx].values[0]
    if aux_data is None:
        porosity = model_param_df['theta_s'][idx].values[0]
        theta_wp = model_param_df['theta_r'][idx].values[0] * top_layer_depth
    else:
        # could be written better (later!)
        porosity = np.nanmax(aux_data['sm_scva'])
        theta_wp = np.nanmin(aux_data['sm_scva']) * top_layer_depth
    theta_s = porosity * top_layer_depth
    theta_star = theta_s * 0.6
    vgm_n = model_param_df['n'][idx].values[0]

    # soil properties for CORDOVA-BRAS VGM model
    return [theta_s, theta_wp, theta_star, k_s, psi_s, vgm_n, porosity, d_i, ps_i, pc_i]



"Cordova model"
def cordova_da(data, theta_i, params):
    """
    Parameters
    ----------
    data: i: Index, 1, p, data['dt'].loc[i], data['et'].loc[i]
    theta_i: Initial soil moisture [mm]
    theta_s: Soil moisture at Saturation [mm]
    theta_wp: Soil moisture at Wilting Point or residual soil moisture [mm]
    porosity: Porosity [%]
    k_s: Hydraulic conductivity at saturation [mm/d]
    psi_s: Saturated soil matric potential [mm]
    d_i: Diffusivity index
    vgm_n: VGM fit parameter
    m: Pore Size index
    pci: Pore connectivity index
    """
    theta_s, theta_wp, theta_star, k_s, psi_s, vgm_n, porosity, d_i, ps_i, pc_i = params
    c = 1 + 2 / (1 - 1/vgm_n)
    m = 2/(c - 3)
    
    if k_s >  0.5 * theta_s:
        k_s = 0.1 * theta_s

    def infilt(theta_i, t_r, i):
        omega = 0
        s_0 = (theta_i)/(theta_s)

        def phi(d_i, s_0):
            def sum_term(n_i):
                return 1/(d_i + 5/3 - n_i) * binom(d_i, n_i) * (s_0/(1 - s_0))**n_i
            sum_all = 0
            for n_i in range(1, d_i + 1):
                sum_all = sum_all + sum_term(n_i)
            return (1 - s_0)**d_i * (1/(d_i + 5/3) + sum_all)

        s = 2 * (1 - (s_0)) * ((5 * porosity * k_s * psi_s *
                                phi(d_i, s_0))/(3 * m * np.pi))**0.5
        a = 0.5 * k_s * (1 + (s_0)**c) - omega
        t_0 = s**2/(2 * (i-a)**2)
        if t_r <= t_0:
            infiltration = i * t_r
        else:
            infiltration = a * t_r + s * np.sqrt(t_r/2)
        return infiltration

    def perc(theta_i):
        # k_s = 30  # Hydraulic conductivity at saturation [mm/d]
        # c = 10  # pore connectivity index
        omega = 0
        # _theta_s = theta_s - theta_wp # Soil moisture at saturation [mm]

        return k_s * (theta_i/theta_s)**c - omega

    def evap(theta_i, et_p):
        # input:
        # theta_i: initial soil moisture at time t
        # Potential ET at time t
        # Output:
        # et_a: Actual ET
        # Parameters:
        k_c = 1      # crop coefficient
        # e_0 = 5          # Another equation to be possibly implemented ? potential evaporative flux
        b = 1.0          # coefficient

        # a = yaron_c# 0.019#k_c * et_p/(theta_star**b)
        a = k_c * et_p/(theta_star**b)

        if (theta_i) >= theta_star:
            et_a = et_p
        else:
            et_a = a * ((theta_i))**b

        return et_a

    t, t_r, pre, dt, et_p = data
    if pre > 0:
        if theta_i >= theta_s:
            ds = 0
        elif infilt(theta_i, t_r, pre) + theta_i- evap(theta_i, et_p) - perc(theta_i)> theta_s:
            ds = theta_s - theta_i 
        else:
            ds = infilt(theta_i, t_r, pre) - evap(theta_i, et_p) - perc(theta_i)
    else:
        ds = - evap(theta_i, et_p) - perc(theta_i)
    theta_i = theta_i + ds
    res_p = (t, theta_i, ds, dt, evap(theta_i, et_p))

    return res_p
