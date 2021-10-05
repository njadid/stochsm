import numpy as np
from scipy.special import binom
import pandas as pd
def sm_model(data, theta_i, theta_s, theta_wp, theta_star, et_p, yaron_c):
    '''data, theta_i, theta_s, theta_wp, theta_star, et_p, yaron_c'''
    def infilt(theta_i, t_r, i):
        # Infiltration rate
        n = 0.35 # Porosity [%]
        k_s = 30 # Hydraulic conductivity at saturation [mm/d]
        psi_s = 190 # Saturated soil matric potential [mm]
        d = 5#.5 # Diffusivity index
        m = 0.286 # Pore Size index
        c = 10 # pore connectivity index
        # pwp = 173 # Permanent Wilting Point [mm]
        # theta_s = 747 # Soil moisture at saturation [mm]
        omega = 0
        s_0 = (theta_i)/(theta_s + theta_wp)
        
        def phi(d, s_0):
            def sum_term(n_i):
                return 1/(d + 5/3 - n_i) * binom(d, n_i) * (s_0/(1 - s_0))**n_i
            sum_all = 0
            for n_i in range(1, d + 1):
                sum_all = sum_all + sum_term(n_i)
            return (1 - s_0)**d * (1/(d + 5/3) + sum_all)

        s = 2 * (1 - (s_0)) * ((5 * n * k_s * psi_s * phi(d, s_0))/(3 * m * np.pi))**0.5
        a = 0.5 * k_s * (1 + (s_0)**c) - omega 
        t_0 = s**2/(2 * (i-a)**2)
        # print(t, a, s)
        if t_r <= t_0:
            infiltration = i * t_r
        else:
            infiltration = a * t_r + s * np.sqrt(t_r/2)
        return infiltration


    def perc(theta_i):
        k_s = 30 # Hydraulic conductivity at saturation [mm/d]
        c = 10 # pore connectivity index
        omega = 0
        # _theta_s = theta_s - theta_wp # Soil moisture at saturation [mm]

        return k_s * (theta_i/theta_s)**c - omega


    def evap(theta_i, et_p):
        ## input:
            # theta_i: initial soil moisture at time t
            # Potential ET at time t
        ## Output:
            # et_a: Actual ET
        ## Parameters:    
        k_c =   0.5      # crop coefficient
        e_0 = 5          # potential evaporative flux
        b = 1.0          # coefficient
        
        a = yaron_c# 0.019#k_c * et_p/(theta_star**b)
        if (theta_i) >= theta_star:
            et_a = et_p
        else:
            et_a = a * ((theta_i))**b
        return et_a


    res_p = []
    res_p.append((0, theta_i, 0, data[0][3], 0))
    for x in data[1:]:
        t, t_r, pre, dt = x
        if pre > 0:
            if theta_i >= theta_s:
                ds = 0
            elif infilt(theta_i, t_r, pre) + theta_i > theta_s:
                ds = theta_s - theta_i
            else:
                ds = infilt(theta_i, t_r, pre)      
        else:
            ds = - evap(theta_i, et_p) - perc(theta_i) 
        theta_i = theta_i + ds 
        res_p.append((t, theta_i, ds, dt, evap(theta_i, et_p)))
        
        
        
    res_p = np.array(res_p)
    data = np.array(data)
    return [res_p, data]



def sm_model_time_based(data, theta_i, theta_s, theta_wp, theta_star, et_p, yaron_c, tscale):
    '''data, theta_i, theta_s, theta_wp, theta_star, et_p, yaron_c'''
    div_fac = {'30min':24*2, 'H':24, 'D':1}
    def infilt(theta_i, t_r, i):
        # Infiltration rate
        n = 0.35 # Porosity [%]
        k_s = 30/div_fac[tscale] # Hydraulic conductivity at saturation [mm/d]
        psi_s = 190 # Saturated soil matric potential [mm]
        d = 5#.5 # Diffusivity index
        m = 0.286 # Pore Size index
        c = 10 # pore connectivity index
        # pwp = 173 # Permanent Wilting Point [mm]
        # theta_s = 747 # Soil moisture at saturation [mm]
        omega = 0
        s_0 = (theta_i)/(theta_s + theta_wp)
        
        def phi(d, s_0):
            def sum_term(n_i):
                return 1/(d + 5/3 - n_i) * binom(d, n_i) * (s_0/(1 - s_0))**n_i
            sum_all = 0
            for n_i in range(1, d + 1):
                sum_all = sum_all + sum_term(n_i)
            return (1 - s_0)**d * (1/(d + 5/3) + sum_all)

        s = 2 * (1 - (s_0)) * ((5 * n * k_s * psi_s * phi(d, s_0))/(3 * m * np.pi))**0.5
        a = 0.5 * k_s * (1 + (s_0)**c) - omega 
        t_0 = s**2/(2 * (i-a)**2)
        # print(t, a, s)
        if t_r <= t_0:
            infiltration = i * t_r
        else:
            infiltration = a * t_r + s * np.sqrt(t_r/2)
        return infiltration


    def perc(theta_i):
        k_s = 30/div_fac[tscale] # Hydraulic conductivity at saturation [mm/d]
        c = 10 # pore connectivity index
        omega = 0
        # _theta_s = theta_s - theta_wp # Soil moisture at saturation [mm]

        return k_s * (theta_i/theta_s)**c - omega


    def evap(theta_i, et_p):
        ## input:
            # theta_i: initial soil moisture at time t
            # Potential ET at time t
        ## Output:
            # et_a: Actual ET
        ## Parameters:    
        k_c =   1      # crop coefficient
        e_0 = 5          # potential evaporative flux
        b = 1.0          # coefficient
        
        # a = yaron_c# 0.019#k_c * et_p/(theta_star**b)
        a = k_c * et_p/(theta_star**b)

        if (theta_i) >= theta_star:
            et_a = et_p
        else:
            et_a = a * ((theta_i))**b
        # et_a = et_p
        return et_a


    res_p = []
    res_p.append((0, theta_i, 0, data[0][3], 0))
    for x in data[1:]:
        t, t_r, pre, dt = x
        if pre > 0:
            if theta_i >= theta_s:
                ds = 0
            elif infilt(theta_i, t_r, pre) + theta_i > theta_s:
                ds = theta_s - theta_i
            else:
                ds = infilt(theta_i, t_r, pre)      
        else:
            ds = - evap(theta_i, et_p) - perc(theta_i) 
        theta_i = theta_i + ds 
        res_p.append((t, theta_i, ds, dt, evap(theta_i, et_p)))
        
        
        
    res_p = np.array(res_p)
    data = np.array(data)
    return [res_p, data]


def sm_model_eagleson(data, soil_params, et_p, yaron_c, tscale):
    '''data, theta_i, theta_s, theta_wp, theta_star, et_p, yaron_c'''
    theta_i, theta_s, theta_wp, theta_star, k_s, psi_s, m = soil_params
    div_fac = {'30min':24*2, 'H':24, 'D':1}
    k_s = k_s/div_fac[tscale]
    
    def infilt(theta_i, t_r, i):
        # Infiltration rate
        n = 0.35 # Porosity [%]
        # k_s = 30/div_fac[tscale] # Hydraulic conductivity at saturation [mm/d]
        # psi_s = 190 # Saturated soil matric potential [mm]
        d = 5#.5 # Diffusivity index
        # m = 0.286 # Pore Size index
        c = 10 # pore connectivity index
        # pwp = 173 # Permanent Wilting Point [mm]
        # theta_s = 747 # Soil moisture at saturation [mm]
        omega = 0
        s_0 = (theta_i)/(theta_s + theta_wp)
        
        def phi(d, s_0):
            def sum_term(n_i):
                return 1/(d + 5/3 - n_i) * binom(d, n_i) * (s_0/(1 - s_0))**n_i
            sum_all = 0
            for n_i in range(1, d + 1):
                sum_all = sum_all + sum_term(n_i)
            return (1 - s_0)**d * (1/(d + 5/3) + sum_all)

        s = 2 * (1 - (s_0)) * ((5 * n * k_s * psi_s * phi(d, s_0))/(3 * m * np.pi))**0.5
        a = 0.5 * k_s * (1 + (s_0)**c) - omega 
        t_0 = s**2/(2 * (i-a)**2)
        # print(t, a, s)
        if t_r <= t_0:
            infiltration = i * t_r
        else:
            infiltration = a * t_r + s * np.sqrt(t_r/2)
        return infiltration


    def perc(theta_i):
        # k_s = 30/div_fac[tscale] # Hydraulic conductivity at saturation [mm/d]
        c = 10 # pore connectivity index
        omega = 0
        # _theta_s = theta_s - theta_wp # Soil moisture at saturation [mm]

        return k_s * (theta_i/theta_s)**c - omega


    def evap(theta_i, et_p):
        ## input:
            # theta_i: initial soil moisture at time t
            # Potential ET at time t
        ## Output:
            # et_a: Actual ET
        ## Parameters:    
        k_c =   1      # crop coefficient
        e_0 = 5          # potential evaporative flux
        b = 1.0          # coefficient
        
        # a = yaron_c# 0.019#k_c * et_p/(theta_star**b)
        a = k_c * et_p/(theta_star**b)

        if (theta_i) >= theta_star:
            et_a = et_p
        else:
            et_a = a * ((theta_i))**b
        # et_a = et_p
        return et_a


    res_p = []
    res_p.append((0, theta_i, 0, data[0][3], 0))
    for x in data[1:]:
        t, t_r, pre, dt = x
        if pre > 0:
            if theta_i >= theta_s:
                ds = 0
            elif infilt(theta_i, t_r, pre) + theta_i > theta_s:
                ds = theta_s - theta_i
            else:
                ds = infilt(theta_i, t_r, pre)      
        else:
            ds = - evap(theta_i, et_p) - perc(theta_i) 
        theta_i = theta_i + ds 
        res_p.append((t, theta_i, ds, dt, evap(theta_i, et_p)))
        
        
        
    res_p = np.array(res_p)
    data = np.array(data)
    return [res_p, data]