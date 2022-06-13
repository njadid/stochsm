import pandas as pd
import numpy as np
import os
from ismember import ismember
pd.set_option('mode.chained_assignment', None)

# Preprocess data to daily time scale
def preprocess(grid_xy):
    """
    Parameters
    -----------
    grid_xy: int
    The grid ID of the SMAP pixel for fetching the soil properties and timeseries of Precipitation and ET
    
    Returns
    -----------
    Daily timescale precip, SMAP soil moisture SCVA, DCA, and ET 
    """
    def export2csv(data_daily):
        fn_out_fmt = '/storage/scratch1/6/njadidoleslam3/gpm_data/daily/{grid_xy}.csv'
        fn_out = fn_out_fmt.format(grid_xy=grid_xy)
        if not os.path.exists(fn_out):
            data_daily.to_csv(fn_out, index=False, float_format='%.3f')

    def read_data(grid_xy):
        prod_list = dict({'gpm': {'vars': ['dt', 'gpm_p']}, 'usgs_pet': {'vars': ['dt', 'et']}})
        fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/smap_gw_data/{year}/{prod}/{grid_xy}.csv'
        data_temp = dict()
        for prod in list(prod_list.keys()):
            data_temp[prod] = pd.DataFrame(columns=prod_list[prod]['vars'])
            for year in range(2001, 2021):
                fn = fn_fmt.format(year=year, prod=prod, grid_xy=grid_xy)
                data_temp[prod] = pd.concat([data_temp[prod], pd.read_csv(fn)])
            data_temp[prod]['dt'] = pd.to_datetime(data_temp[prod]['dt'])
            data_temp[prod].sort_values(by='dt').reset_index()
        return data_temp

    def convert2daily(data):
        start_dt =  '1/1/2001'
        end_dt =    '12/31/2020'

        def create_base_df():
            ts_series = pd.date_range(start=start_dt, end=end_dt)
            return pd.DataFrame({'dt':ts_series})

        def gapfill_et(raw_et):
            def get_daily_average():
                et_average = pd.DataFrame()
                et_average['doy'] = np.arange(1,367)
                et_average['mean_et'] = raw_et['et'].groupby([(raw_et['dt'].dt.month),(raw_et['dt'].dt.day)]).mean().values
                return et_average

            
            et_average = get_daily_average()
            
            raw = pd.merge(base_df, raw_et, on='dt', how='left')
            # raw = raw.reset_index()
            bool_doy, idx_doy = ismember(raw['dt'].dt.dayofyear.values, et_average['doy'].values)
            idx_nan = (raw['et'].isna()) & bool_doy
            idx_doy = idx_doy[idx_nan]
            raw['et'].iloc[idx_nan] = et_average['mean_et'][idx_doy]
            return raw

        base_df = create_base_df()
        data['gpm'] = data['gpm'].set_index('dt').copy(deep=True)
        gpm_daily = data['gpm'].resample('D').sum()
        data_daily = pd.merge(base_df, gpm_daily, on='dt', how='left')

        gap_filled_et = gapfill_et(data['usgs_pet'])
        data_daily = pd.merge(data_daily, gap_filled_et, on='dt', how='left')
        
        
        data_daily = data_daily.sort_values(by='dt')
        # data_daily.drop('index', axis=1, inplace=True)
        data_daily['gpm_p'].loc[data_daily['gpm_p'] <0] = 0
        data_daily.replace(-9999, 0, inplace=True)
        data_daily = data_daily.astype({'et': 'float64', 'gpm_p':'float64'})
        return data_daily

    data_temp = read_data(grid_xy)
    data_daily = convert2daily(data_temp)
    # export2csv(data_daily)
    return data_daily
