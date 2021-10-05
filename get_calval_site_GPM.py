# lat_lons = [(row['lat'], row['lon']) for i,row in enumerate(calval_sites)]
import pandas as pd
import h5py
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import numpy as np
import os
calval_sites = pd.read_csv('/storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/data/calval_sites.csv', delimiter=',')

def get_grid_x_y(in_lat, in_lon):
    def geo_idx(dd, dd_array):
        """
        search for nearest decimal degree in an array of decimal degrees and return the index.
        np.argmin returns the indices of minium value along an axis.
        so subtract dd from all values in dd_array, take absolute value and find index of minium.
        """
        geo_idx = (np.abs(dd_array - dd)).argmin()
        return geo_idx

    fn = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06/2000/366/3B-HHR.MS.MRG.3IMERG.20001231-S233000-E235959.1410.V06B.HDF5'
    dset = h5py.File(fn,'r')
    lats  = dset['Grid/lat'][:]
    lons  = dset['Grid/lon'][:]

    lat_idx = geo_idx(in_lat, lats)
    lon_idx = geo_idx(in_lon, lons)
    return lat_idx, lon_idx

calval_sites = pd.read_csv('/storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/data/calval_sites.csv', delimiter=',')
fn_temp  = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/time/{year}/{month}.h5'
path_out_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/calval_sites/{year}/{month}/'
fn_out_fmt = '{site_id}.csv'
def get_data(year):
    for month in range(5,13):
        out_path = path_out_fmt.format(year = year, month=str(month).zfill(2))
        if not os.path.exists(out_path):
            os.makedirs(out_path)


        fn = fn_temp.format(year = year, month=str(month).zfill(2))
        f = h5py.File(fn,'r')
        dset = f['precipitationCal']
        
        time = f['time'][:]
        t = time.reshape((int(time.shape[0]/2),2))[:,0]
        dt = pd.to_datetime(t, unit='s')
        for i in range(len(calval_sites)):
            in_lat = calval_sites.loc[i]['lat']
            in_lon = calval_sites.loc[i]['lon']
            lat_idx, lon_idx = get_grid_x_y(in_lat, in_lon)
            p_data = dset[:, lon_idx,lat_idx ]
            p_data =  p_data.reshape((int(p_data.shape[0]/2),2)).mean(axis=1)
            fn_out = os.path.join(out_path, fn_out_fmt.format(site_id = calval_sites.loc[i]['Site ID']))
            df = pd.DataFrame({'dt':dt, 'p':p_data }).to_csv(fn_out, index=False)
if __name__=='__main__':
    n_cpu = cpu_count()
    print('# of CPUS working: ' + str(n_cpu))
    pool = ThreadPool(n_cpu)
    permute_list = list([x for x in range(2015,2021)])
    pool.map(get_data, permute_list)