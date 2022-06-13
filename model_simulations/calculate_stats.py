from zmq import SUBSCRIBE
from model_engine import cordova_vgm_params, cordova_da
import numpy as np
import pandas as pd
import datetime
import os

def calc_stats(data_daily, s_props, grid_xy, run_id):

    def forcing_prep(forcings):
        """
        Parameters
        -----------
        data: DataFrame
            Original precip data[mm/day], Soil Moisture [m3/m3], ET [mm/day]
        """
        data_forcing = []
        for i, p in enumerate(forcings['gpm_p']):
            if p > 0:
                data_forcing.append(
                    (i, 1, p, forcings['dt'].iloc[i], forcings['et'].iloc[i]))
            else:
                data_forcing.append(
                    (i, 0, 0, forcings['dt'].iloc[i], forcings['et'].iloc[i]))
        return data_forcing


    def cb2volsm(x_array):
        return (x_array + theta_wp)/(top_layer_depth + theta_wp)


    def set_inits(theta_wp, theta_s):
        init_array = np.linspace(0, 1, 41, endpoint=True, dtype='float16') * top_layer_depth
        idx_init_conds = (init_array > theta_wp) & (init_array < theta_s)
        
        return init_array[idx_init_conds]
    

    def openloop(data_forcing, theta_i1, params):
        sim_data = [theta_i1]
        n_data = len(data_forcing)
        for i in range(n_data):
            if i == 0:
                theta_i = theta_i1
            else:
                theta_i = sm_model
            forcing_at_t = list(data_forcing[i])
            res_p = cordova_da(forcing_at_t, theta_i, params)
            sm_model = res_p[1]
            sim_data.append(sm_model)
        # sim_data = cb2volsm(np.array(sim_data))
        sim_data = cb2volsm(np.array(sim_data, dtype=np.float16))
        return sim_data
    
    def stage_indices(month, ds):
        if month ==12:
            indices = [x for x in range(8*ds-1, 720, 36)]
            indices[-1] = 695
        else:
            indices = [x for x in range(8*ds-1, 720, 36)]
        return np.array(indices, dtype=int)

    def subset_data(data_daily, year, month):
        dt_start =  pd.Timestamp(datetime.date(year, month, 1))
        dt_end =    pd.Timestamp(datetime.date(year, month, 1) + datetime.timedelta(days = 35))
        idx = (data_daily['dt'] >= dt_start) & (data_daily['dt'] < dt_end)
        return data_daily.loc[idx]
     

    def run_model():
        results = dict()
        for month in range(1,13):
            results[month] = dict()
            for theta_i1 in init_conds:
                results[month][theta_i1] = []
                for year in range(2001, 2021):
                    filtered = subset_data(data_daily, year, month)
                    subset_forcing = forcing_prep(filtered)                
                    _sm_array = openloop(subset_forcing, theta_i1, params)
                    # return subset_forcing
                    
                    results[month][theta_i1].extend(_sm_array)
        return results
    
    def postprocess_results(results):
        fn_out_fmt =  out_path + '{grid_xy}.csv'
        _data = []
        for month in range(1,13):
            for ds in range(1,5):
                init_list = results[month].keys()
                for init in init_list:
                    idx = stage_indices(month=month, ds=ds)
                    sm_list = np.array(results[month][init])
                    sm_list = sm_list[idx]
                    _data.append([month, ds,  cb2volsm(init), np.nanmean(sm_list), np.nanstd(sm_list)])

        out_df = pd.DataFrame(_data, columns = ['month', 'stage', 'init', 'mean', 'sd'])
        out_fn = fn_out_fmt.format(grid_xy = grid_xy )
        out_df.to_csv(out_fn, float_format='%.4f', index = False)
        # return out_df

    out_path_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_results_v0/{run_id}/'
    out_path = out_path_fmt.format(run_id = str(run_id))
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # outer scope variables.
    top_layer_depth = 1000
    params = cordova_vgm_params(s_props, grid_xy, top_layer_depth=top_layer_depth, aux_data=None)
    theta_s = params[0] # model sat sm
    theta_wp = params[1] # model wilting point sm

    init_conds = set_inits(theta_wp, theta_s)
    results = run_model()
    postprocess_results(results)
    # return out
    # theta_i = theta_s/2
    # subset_forcing = forcing_prep(data_daily)                
    # _sm_array = openloop(subset_forcing, theta_i, params)
    # return _sm_array
    