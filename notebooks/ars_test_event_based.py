import pandas as pd
import numpy as np
# import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import datetime
import pickle
from functions.event_detector import storm_def

from functions.check_model import sm_model_eagleson

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('font', family='default')


## some convbersion factor based on time-scale of the forcings
tscale = 'H'
div_fac = {'30min':24*2, 'H':24, 'D':1}


## load data 
with open('/mnt/d/ubuntu/projects/gatechProjects/StochSM/data/gpm_gage_sf.pickle', 'rb') as handle:
    gpm_gage_sf = pickle.load(handle)


# load soil properties for the model simulations 
s_props = pd.read_csv('/mnt/d/ubuntu/projects/gatechProjects/StochSM/data/gages/sm_gages_w_SProps.csv')

## metadata for organization of the soil moisture results from the model 
prod_list = dict({
                'tscales': ['H'],
                'scale_f': 1,
                'dt_col': 'date',
                'p_cols': ['value', 'PREC_1', 'PREC_2'],
                'p_thres': 0,
                'alias': ['GPM', 'Gage_1', 'Gage_2']
                 })

## match the GPM and in situ gage observations of rainfall 
gage_gpm_matched = dict()
for i, st_id in enumerate(gpm_gage_sf['lut_gpm_gage']['st_id'][0:15]):
    fid = str(gpm_gage_sf['lut_gpm_gage']['FID'][i])
    st_id = str(st_id)
    gpm_gage_ = gpm_gage_sf['gpm_gage'][fid]
    sf_gages_ = gpm_gage_sf['sf_gages'][st_id]
    gage_gpm_matched[st_id] = pd.merge(gpm_gage_, sf_gages_)

## rainfall event detection and assortment
mit = 9
p_events = dict()
gage_gpm_event_based = dict()
for i, st_id in enumerate(gpm_gage_sf['lut_gpm_gage']['st_id'][0:15]):
    st_id = str(st_id)
    p_new =  dict()
    for p_col in prod_list['p_cols']:
        p_new[p_col] = []
        mit_dry = []
        data_ = gage_gpm_matched[st_id][['date', p_col]].replace(np.nan, 0)
        events = storm_def(data_, mit, dt_col='date', p_col=p_col)
        dry_periods = [(events['start'][i+1] - events['end'][i]).total_seconds()/3600.0 for i in range(len(events)-1)]
        storm_arrivals = [(events['start'][i+1] - events['start'][i]).total_seconds()/3600.0 for i in range(len(events)-1)]
        mit_dry.append((mit, np.std(dry_periods)/ np.mean(dry_periods),np.std(storm_arrivals)/ np.mean(storm_arrivals), 
        np.mean(dry_periods), np.mean(storm_arrivals)))
        p_events = np.array(mit_dry)
        for event in events.itertuples():
            for n in range(int(event.deltat)):
                p_new[p_col].append((event.start+datetime.timedelta(hours=n), event.precip))
        p_new[p_col] = pd.DataFrame(p_new[p_col], columns=['date',p_col])
        p_new[p_col] = p_new[p_col].set_index('date')
        dates = pd.date_range(gage_gpm_matched[st_id]['date'].iloc[0], gage_gpm_matched[st_id]['date'].iloc[-1], freq='H')
        p_new[p_col] = p_new[p_col].reindex(dates).fillna(0)
        p_new[p_col]= p_new[p_col].reset_index().rename(columns={'index':'date'})
        # p_new[p_col] = p_new.reset_index()
    gage_gpm_event_based[st_id] = pd.DataFrame({'date':p_new['value']['date'],
'value':p_new['value']['value'],'PREC_1': p_new['PREC_1']['PREC_1'], 'PREC_2': p_new['PREC_2']['PREC_2']})
    
        # p_new[p_col].set_index('date',inplace=True)
    # gage_gpm_event_based[st_id] = pd.merge([p_new['value'], p_new['PREC_1'], p_new['PREC_2']],on = 'date').reset_index()

## Dictionary storing the results for the soil moisture and raionfall observations (GPM and dual tipping bucket raingage)
results = dict()

# loop over stations
for st_id in gpm_gage_sf['lut_gpm_gage']['st_id'][0:15]:
    st_id = str(st_id)
    
    idx = np.where(s_props['name']=='ARS' + st_id)
    
    if len(idx[0]) ==0:
        idx = 61 
    else:
        idx = idx[0][0]

    # prescribed constant daily evapotranspiration
    et_p = 6.3
    
    # soil properties from POLARIS soil dataset (doi:10.1029/2018WR022797)
    theta_s = s_props['theta_s'][idx] # cm3/cm3
    theta_s_mm = theta_s * 1000 # cm^3/cm^3 to mm in a reference one-meter soil column
    theta_i = theta_s_mm # Initializing the soil moisture as a saturated condition # this can be changed 
    theta_wp = 0.3 * theta_s_mm # 0.3 comes from ratio of the theta_WP/theta_s from cordova-bras 1981 (doi:10.1029/WR017i001p00093)
    theta_star = 330/574 * theta_s_mm  # Field Capacity # similarly, to relate the field capacity to soil moisture saturation (unit :mm)
    k_s = 10**(s_props['ksat'][idx])*24 * 10 # cm/hr -> cm/day -> mm/day
    psi_s = 10**(s_props['hb'][idx]) * 101.97 # kPa -> mm
    m = s_props['lambda'][idx]  # Brooks-Corey pore size distribution index [-]
    yaron_c = 0.022  # havent figured this one out yet need more study
    
    soil_params = [theta_i, theta_s_mm, theta_wp, theta_star, k_s, psi_s, m]

    results[st_id] = dict()
    for p_obs in prod_list['p_cols']:
        results[st_id][p_obs] = dict()
        gage_gpm_event_based[st_id][p_obs] = gage_gpm_event_based[st_id][p_obs].replace(np.nan,0)
        data_forcing = []
            
        for i, p in enumerate(gage_gpm_event_based[st_id][p_obs]):
            
            if p>0:
                data_forcing.append((i, 1, p*24, gage_gpm_event_based[st_id]['date'].loc[i]))
            else:
                data_forcing.append((i, 0, 0, gage_gpm_event_based[st_id]['date'].loc[i]))

        results[st_id][p_obs]['res_p'], results[st_id][p_obs]['data_p'] = sm_model_eagleson(data_forcing, soil_params, et_p/div_fac[tscale], yaron_c, tscale)
        results[st_id][p_obs]['res_p'][:,1] = results[st_id][p_obs]['res_p'][:,1]/theta_s_mm*theta_s
    results[st_id]['SM'] = pd.DataFrame({'date':gage_gpm_event_based[st_id]['date'], 
            'SM_5':gage_gpm_matched[st_id]['SM_5'],
            'SM_10': gage_gpm_matched[st_id]['SM_10'],
            'SM_20': gage_gpm_matched[st_id]['SM_20'],
            'SM_50': gage_gpm_matched[st_id]['SM_50'],
            'SM_avg':np.nanmean([gage_gpm_matched[st_id]['SM_5'] ,
                                gage_gpm_matched[st_id]['SM_10'] ,
                                gage_gpm_matched[st_id]['SM_20'] ,
                                gage_gpm_matched[st_id]['SM_50'] ], axis=0)}
                                ,dtype=object)


with open('/mnt/d/ubuntu/projects/gatechProjects/StochSM/data/gages/ARS_data/gpm_gage_sm_event_based.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol= pickle.HIGHEST_PROTOCOL)
