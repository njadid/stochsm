import sys
import importlib
import pandas as pd
import numpy as np
import datetime
import os
#%%

class get_Forcings:
    in_path = '/mnt/d/ubuntu/projects/gatechProjects/StochSM/data/gages'
    path_template = 'unifieddata/{0}/{1}.csv'
    @classmethod
    def get_HLM_254_forcings(cls, st_name, year):
        def getPrecip(st_name, year):
            fn_in = os.path.join(cls.in_path, cls.path_template.format(year, st_name))
            data = pd.read_csv(fn_in, parse_dates=["dt"], index_col="dt")
            data_new = data.resample('1min').first().ffill()
            data_new['p_1'] = data['p_1'].resample('1min').first().ffill()
            data_new['p_2'] = data['p_2'].resample('1min').first().ffill()
            data_new = data_new.fillna(value=0)
            return data_new

        def getPET(year):
            et = pd.read_csv(os.path.join(cls.in_path, 'evap.csv'))
            et['month'] = [datetime.datetime(month=mnth, day=1, year=year) for mnth in et['month']]
            et = et.set_index('month')
            start_date = et.index.min() - pd.DateOffset(day=1)
            end_date = et.index.max() + pd.DateOffset(day=31)
            dates = pd.date_range(start_date, end_date, freq='min')
            dates.name = 'dt'
            return et.reindex(dates, method='ffill')
        # df['pet'] = np.divide(df['pet'],float(30.0*24*60))
        gage_data = getPrecip(st_name, year)
        pet = getPET(year)
        gage_data = pd.merge(gage_data, pet, how='inner', on='dt')
        gage_data = gage_data.loc[(gage_data.index.month > 3) & (gage_data.index.month < 11)]
        forcings = [gage_data['p_1'].values, gage_data['pet'].values] # mm/15min -> mm/hr
        return gage_data, forcings

    @classmethod
    def get_herrada_forcings(cls, st_name, year):
        def getPrecip(st_name, year):
            fn_in = os.path.join(cls.in_path, cls.path_template.format(year, st_name))
            data = pd.read_csv(fn_in, parse_dates=["dt"], index_col="dt")
            data_new = data.resample('60min').first()
            data_new['p_1'] = data['p_1'].resample('60min').sum()
            data_new['p_2'] = data['p_2'].resample('60min').sum()
            data_new = data_new.fillna(value=0)
            return data_new
        def getPET(year):
            et = pd.read_csv(os.path.join(cls.in_path, 'evap.csv'))
            et['month'] = [datetime.datetime(month=mnth, day=1, year=year) for mnth in et['month']]
            et = et.set_index('month')
            start_date = et.index.min() - pd.DateOffset(day=1)
            end_date = et.index.max() + pd.DateOffset(day=31)
            dates = pd.date_range(start_date, end_date, freq='H')
            dates.name = 'dt'
            return et.reindex(dates, method='ffill')

        gage_data = getPrecip(st_name, year)
        pet = getPET(year)
        pet = np.divide(pet, float(30.0 * 24))
        gage_data = pd.merge(gage_data, pet, how='inner', on='dt')
        gage_data = gage_data.loc[(gage_data.index.month > 3) & (gage_data.index.month < 11)]
        forcings = [gage_data['p_1'].values * 0.1, gage_data['pet'].values * 0.1] # mm/hr -> cm/hr
        return gage_data, forcings