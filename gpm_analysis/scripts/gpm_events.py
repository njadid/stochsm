import sys
import gzip
import numpy as np
import pandas as pd
import pickle

def read_gz(fn):   
    # fn = '/storage/coda1/p-rbras6/0/njadidoleslam3/persiann/persiann.eng.uci.edu/CHRSdata/PERSIANN/hrly/2006/m6s4rr0606100.bin.gz'
    try:
        f=gzip.GzipFile(fn)
        file_content = f.read()
        array = np.frombuffer(file_content, dtype='>f4')
        return array.reshape(480,1440)
    except:
        return np.ones((480, 1440))* -9999.0


def create_dt_vec(year):
    # returns date vector as 
    # 1- date string with %yy%DOY%H 2- a vector of the months
    start_dt = '{year}-1-1'.format(year = year)
    end_dt = '{year}-1-1'.format(year = year + 1)
    dt_vec = pd.date_range(start=start_dt, end=end_dt, freq='60min',closed='left')
    return dt_vec.strftime('%y%j%H').values, dt_vec.month


def extract_events1(p_data):
    # %%timeit
    def storm_def():
        def calc_min_mit_new(p_events):
            idx, = np.where((np.diff(np.sign(p_events[:, 1]-1)) != 0)*1 == 1)
            if len(idx) == 0:
                a = np.abs(p_events[:, 1]-1)
                idx, = np.where(a == a.min())
                return p_events[:, 0][idx][0]
            return (p_events[:, 0][idx][0]+p_events[:, 0][idx+1][0])/2.0


        def check_mits_new(mit):
            idx = (bin_events_month <= -mit)*(bin_events_month < 0)
            dry_periods = bin_events_month[idx]*-1
            return np.array([mit, np.std(dry_periods[:]) / np.mean(dry_periods[:]),
                            np.int32(np.mean(dry_periods[:]))])


        # One-time operations for dry-wet periods        
        p_data[-1] = 1
        v = (p_data == 0)*1
        n = v == 0
        a = ~n
        c = np.cumsum(a)
        d = np.diff(np.append([0.], c[n]))
        v[n] = -d
        dry_vec = np.cumsum(v)
        bin_events = np.append([1], np.diff(dry_vec))

        event_idx = bin_events < 0

        # Initialize the result array
        result = []

        # Loop over each month
        for m in range(1,13):
            mnth_idx = dt_vec == m
            bin_events_month = bin_events[event_idx & mnth_idx]
            # CONDITION 1: If no events, skip the month
            if (len(bin_events_month) == 0):
                # DO NOTHING
                continue

            # CONDITION 2: For Jan - November
            if m < 12:
                idx_next_events = np.where((dt_vec > m) & (event_idx))

                bin_events_next_events = bin_events[idx_next_events]
                bin_events_month = np.append(
                    bin_events_month, bin_events[idx_next_events[0][0]])

                mit_dry = []
                for mit in mit_list:
                    _mit = check_mits_new(mit)
                    mit_dry.append(_mit)
                    if ((_mit[1] < 1.0) & (len(mit_dry) > 1)):
                        break    # condition satisfied

                mit_dry = np.array(mit_dry)
                min_mit = calc_min_mit_new(mit_dry)

                idx = (bin_events <= -min_mit) * (bin_events < 0) * (dt_vec == m)
                date_idx = np.where(idx)[0]

                date_idx_month = np.append(date_idx, idx_next_events[0][0])
                dry_periods = bin_events[date_idx_month]*-1

                event_durations = date_idx_month[1:] - \
                    dry_periods[1:] - date_idx_month[0:-1]
                event_indices = date_idx_month[0:-1] + event_durations
                n_events = len(event_durations)

                p_totals = [0]*n_events
                for i in range(n_events):
                    p_totals[i] = np.sum(p_data[date_idx[i]:event_indices[i]+1])
                result.append([[m]*n_events, event_durations,
                            dry_periods[0:-1], p_totals/event_durations])
            
            # CONDITION 3: For December
            if m == 12:
                if (len(bin_events_month) == 1):
                    continue
                mit_dry = []
                for mit in mit_list:
                    _mit = check_mits_new(mit)
                    mit_dry.append(_mit)
                    if ((_mit[1] < 1.0) & (len(mit_dry) > 1)):
                        break    # condition satisfied
                mit_dry = np.array(mit_dry)
                min_mit = calc_min_mit_new(mit_dry)

                idx = (bin_events <= -min_mit) * (bin_events < 0) * (dt_vec == m)
                date_idx_month = np.where(idx)[0]
                dry_periods = bin_events[date_idx_month]*-1

                date_idx = np.where(idx)[0]
                event_durations = date_idx[1:] - dry_periods[1:] - date_idx[0:-1]
                event_indices = date_idx[0:-1] + event_durations
                n_events = len(event_durations)


                p_totals = [0]*n_events
                for i in range(n_events):
                    p_totals[i] = np.sum(p_data[date_idx[i]:event_indices[i]+1])

                result.append([[m]*n_events, event_durations,
                            dry_periods[0:-1], p_totals/event_durations])

        return result
    #######################

    n_data = len(p_data)
    pos_idx = np.sum((p_data>=0) & (p_data<65535))
        
    if (float(np.sum(pos_idx)/n_data)==0):
        events = []
        # print('No data 2')
    else:
        try:  
            p_data[p_data<0] = 0.0
            events = storm_def()
        except:
            events = []
    return events

######### INPUTS ##########
year = int(sys.argv[1])
# year = 2001
######### INPUTS ##########

######### Global Variables #########
## Formatting variables

out_pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/persiann_analysis/events_mbased/yearly/{year}.pickle'
fn_out_pickle = out_pickle_fmt.format(year=year)

#### variables for defining the events
set_min_mit = 3
mit_list = [x for x in range(set_min_mit,12*24, 6)]
####

### Date vector converted to month index array
fn_list, dt_vec = create_dt_vec(year)
data = np.ones((480,1440, len(fn_list)))

for i, fn in enumerate(fn_list):
    fn1 = '/storage/coda1/p-rbras6/0/njadidoleslam3/persiann/persiann.eng.uci.edu/CHRSdata/PERSIANN/hrly/{year}/m6s4rr{fn}.bin.gz'.format(fn=fn, year = year)
    data[:,:,i] = read_gz(fn1)
# 5 minutes up to here

npy_out = '/storage/coda1/p-rbras6/0/njadidoleslam3/persiann/yearly/{year}.npy'.format(year = year)
with open(npy_out, 'wb') as f:
    np.save(f,data, allow_pickle=True)

grid_y_list = list(range(480))
grid_x_list = list(range(1440))
######### Global Variables #########

with open(fn_out_pickle, 'wb') as handle:
    
    for grid_y in grid_y_list:
        # data = np.array(f.variables['p01m'][:, grid_y , :].data)
        lol = []
        for grid_x in grid_x_list:
            gid = int('1{gid_x}{gid_y}'.format(gid_x = str(grid_x).zfill(4), gid_y = str(grid_y).zfill(4)))
            events = extract_events1(data[grid_y, grid_x, :])
            lol.append([gid,year,] + events)
        pickle.dump(lol, handle, protocol= pickle.HIGHEST_PROTOCOL)


