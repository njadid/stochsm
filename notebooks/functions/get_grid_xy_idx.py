import numpy as np
import h5py
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

in_lat, in_lon = [30.5,56.1]
lat_idx, lon_idx = get_grid_x_y(in_lat, in_lon)