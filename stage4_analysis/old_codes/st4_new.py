
import sys
from netCDF4 import Dataset
import pickle
import numpy as np
import itertools

#######################

def extract_events(p_data):
    def storm_def_new(p_data, mit):
        v = (p_data==0)*1
        n = v==0
        a = ~n
        c = np.cumsum(a)
        d = np.diff(np.append([0.], c[n]))
        v[n] = -d
        dry_vec = np.cumsum(v)
        # mit = 6
        aa = np.append([1], np.diff(dry_vec))
        idx = (aa  <=-mit)*(aa<0)
        dry_periods = aa[idx]*-1
        return np.array([mit, np.std(dry_periods[1:]) / np.mean(dry_periods[1:]), np.int32(np.mean(dry_periods[1:]))])

    #######################

    # @jit('(f4[:],int32)')
    def calc_min_mit(p_events):
        idx, = np.where((np.diff(np.sign(p_events[:,1]-1)) != 0)*1==1)
        if len(idx)==0:
            a = np.abs(p_events[:,1]-1)
            idx, = np.where(a == a.min())
            return p_events[:,0][idx][0]
        return (p_events[:,0][idx][0]+p_events[:,0][idx+1][0])/2.0

    #######################

    def get_events(p_data,set_min_mit):
        mit_list = [x for x in range(set_min_mit,12*24, 6)]
        mit_dry = []
        for mit in mit_list:
            mit_dry.append(storm_def_new(p_data, mit))
            
        mit_dry = np.array(mit_dry)
        min_mit = calc_min_mit(mit_dry)
        event_metrics = storm_def_new(p_data, min_mit)
        # print(event_metrics)
        return event_metrics
    
    
    set_min_mit = 3
    
    
    result = np.empty((1,1), dtype=object)
    
    n_data = len(p_data)
    # event_metrics = get_events(p_data, set_min_mit)
    # %timeit np.sum(p_data < 0)
    pos_idx = np.sum((p_data>=0) & (p_data<65535))
    # neg_idx = ~pos_idx
    # if float(np.sum(neg_idx)/n_data)<0.1:
    #     print('No data 1', float(np.sum(p_data < 0)/n_data))
    p_data[p_data<0] = 0
        
    if (float(np.sum(pos_idx)/n_data)==0):
        result = np.array([np.nan,np.nan,np.nan])
        # print('No data 2')
    else:
        try:
            event_metrics = get_events(p_data, set_min_mit)
            result = event_metrics
        except:
            result = np.array([np.nan,np.nan,np.nan])
            # print('Try-except')


    return result



fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/precipitation/stage4/{year}_stage4_hourly.nc'
out_pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events/{year}_new.pickle'
year = sys.argv[1]

fn_nc_in = fn_fmt.format(year=year)

f = Dataset(fn_nc_in)

data = f.variables['p01m'][:, :, :].data
a = np.apply_along_axis(extract_events, 0, data)
fn_out_pickle = out_pickle_fmt.format(year=year)
with open(fn_out_pickle, 'wb') as handle:
    pickle.dump(a, handle, protocol= pickle.HIGHEST_PROTOCOL)