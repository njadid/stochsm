import sys
import netCDF4
import pickle
import numpy as np
import pandas as pd

def extract_events1(p_data):
    def storm_def():
        def check_mits(mit):
            idx = (bin_events  <=-mit)*(bin_events<0)
            dry_periods = bin_events[idx]*-1
            return np.array([mit, np.std(dry_periods[1:]) / np.mean(dry_periods[1:]),
                                np.int32(np.mean(dry_periods[1:]))])

        v = (p_data==0)*1
        n = v==0
        a = ~n
        c = np.cumsum(a)
        d = np.diff(np.append([0.], c[n]))
        v[n] = -d
        dry_vec = np.cumsum(v)
        bin_events = np.append([1], np.diff(dry_vec))

        
        mit_dry = []
        for mit in mit_list:
            _mit = check_mits(mit)
            mit_dry.append(_mit)
            if ((_mit[1]<1.0) & (len(mit_dry)>1)):
                break    # condition satisfied
        mit_dry = np.array(mit_dry)
        min_mit = calc_min_mit(mit_dry)
        
        idx = (bin_events  <= -min_mit) * (bin_events < 0)
        dry_periods = bin_events[idx]*-1
        date_idx = np.where(idx[1:])[0]
        event_durations = date_idx[1:] -dry_periods[1:] - date_idx[0:-1]
        event_indices = date_idx[0:-1] + event_durations
        n_events = len(event_durations)
        # storm_def_new(data, mit)

        p_totals = [0]*n_events
        for i in range(n_events):
            p_totals[i] = np.sum(p_data[date_idx[i]:event_indices[i]+1])

        
        # summary = pd.DataFrame({'dt':dt_vec[date_idx[1:]], 't0':event_durations, 'tb':dry_periods[0:-1], 'intnesity':p_totals/event_durations})
        summary = [dt_vec[date_idx[0:-1]], event_durations, dry_periods[0:-1], p_totals/event_durations]
        # summary = [dt_vec[date_idx[1:]], event_durations, dry_periods[0:-1], p_totals/event_durations]
        return min_mit, summary
        # return np.array([min_mit, np.std(dry_periods[1:]) / np.mean(dry_periods[1:]), np.int32(np.mean(dry_periods[1:]))])

    #######################


    def calc_min_mit(p_events):
        idx, = np.where((np.diff(np.sign(p_events[:,1]-1)) != 0)*1==1)
        if len(idx)==0:
            a = np.abs(p_events[:,1]-1)
            idx, = np.where(a == a.min())
            return p_events[:,0][idx][0]
        return (p_events[:,0][idx][0]+p_events[:,0][idx+1][0])/2.0

    #######################

    result = np.empty((1,1), dtype=object)   
    n_data = len(p_data)
    # event_metrics = get_events(p_data, set_min_mit)
    # %timeit np.sum(p_data < 0)
    pos_idx = np.sum((p_data>=0) & (p_data<65535))

    # if float(np.sum(neg_idx)/n_data)<0.1:
    #     print('No data 1', float(np.sum(p_data < 0)/n_data))
    # p_data[p_data<0.5] = 0
        
    if (float(np.sum(pos_idx)/n_data)==0):
        result = np.array([np.nan,np.nan,np.nan])
        print('No data 2')
    else:
        # try:
        event_metrics = storm_def()
        result = event_metrics
        # except:
            # result = np.array([np.nan,np.nan,np.nan])
            # print('Try-except')

    # with open(fn_out_pickle, 'wb') as handle:
    #     pickle.dump(result, handle, protocol= pickle.HIGHEST_PROTOCOL)
    return result

######### INPUTS ##########

year = int(sys.argv[1])

######### Global Variables #########

## Formatting variables
# fn_fmt = '/home/navid/Downloads/{year}_stage4_hourly.nc'
fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/precipitation/stage4/{year}_stage4_hourly.nc'
out_pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events/{year}_new.pickle'
# out_pickle_fmt = '/storage/home/hcoda1/6/njadidoleslam3/p-rbras6-0/projects/stochsm/stage4_analysis/events/{year}.pickle'

year1= year+1
start_dt = '{year}-1-1'.format(year = year)
end_dt = '{year}-1-1'.format(year = year + 1)
fn_nc_in = fn_fmt.format(year=year)
fn_out_pickle = out_pickle_fmt.format(year=year)
# technical variables
set_min_mit = 3
mit_list = [x for x in range(set_min_mit,12*24, 6)]


dt_vec = pd.date_range(start=start_dt, end=end_dt, freq='60min',closed='left')
dt_vec = dt_vec.month

# read dataset row by row, i.e., 
f = netCDF4.Dataset(fn_nc_in)
# f.set_auto_maskandscale(False)
# f.set_auto_mask(True)

grid_y_list = list(range(881))
grid_x_list = list(range(1121))

with open(fn_out_pickle, 'wb') as handle:
    
    for grid_y in grid_y_list:
        data = np.array(f.variables['p01m'][:, grid_y , :].data)
        lol = []
        for grid_x in grid_x_list:
            gid = int('1{gid_x}{gid_y}'.format(gid_x = str(grid_x).zfill(4), gid_y = str(grid_y).zfill(4)))
            events = extract_events1(data[:,grid_x])
            lol.append((gid,year,) + events)
            # lol[gid] = (year,) + events
        pickle.dump(lol, handle, protocol= pickle.HIGHEST_PROTOCOL)
