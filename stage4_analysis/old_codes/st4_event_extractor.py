from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
import pickle
import datetime
import numpy as np
import pandas as pd
import sys
import itertools
import datetime
import xarray as xar
# from multiprocessing.dummy import Pool as ThreadPool
# from multiprocessing import cpu_count

def storm_def(data, mit, dt_col, p_col):
    # print(data)
    numtotal = data.shape[0]

    # Define minimum interevent time (MIT), hours:
    # mit = 5

    # Initialize last heard tracker & rain event counter:
    lastheard = data[dt_col][0] - datetime.timedelta(hours=mit+1)
    rainevent = 1
    Event = dict()

    # Filter rainfalls and create storm event objects:
    for i in range(1, numtotal):
        Event[rainevent] = dict()
        Event[rainevent][dt_col] = []
        Event[rainevent][p_col] = []

        if data[p_col][i] > 0 and data[p_col][i-1] == 0:
            # This is the start of a rain event!

            # Save the start time for the storm event:
            stormstart = data[dt_col][i]

            # Check to see if the MIT requirement is met:
            delta_t = (stormstart - lastheard).total_seconds()/3600.0

            if delta_t <= mit:
                # Found a short break in the storm; still in previous event:
                rainevent -= 1

                # Include the starting zero event:
                tprev = data[dt_col][i-1]
                rprev = data[p_col][i-1]
                if  Event[rainevent][dt_col][-1] != tprev:
                    Event[rainevent][dt_col].append(data[dt_col][i-1])
                    Event[rainevent][p_col].append(data[p_col][i-1])
                    

                # Initialize the current rain events rainfall amount:
                eventrain = data[p_col][i]

                # Initialize the iterater:
                j = 0

                while (eventrain > 0):
                    # Save the time and rain amounts for the event:
                    Event[rainevent][dt_col].append(data[dt_col][i+j])
                    Event[rainevent][p_col].append(data[p_col][i+j])

                    # Increment the iterater & update event rainfall:
                    j += 1
                    eventrain = data[p_col][i+j]

                # Include the ending zero event:
                Event[rainevent][dt_col].append(data[dt_col][i+j])
                Event[rainevent][p_col].append(data[p_col][i+j])

                # Update lastheard date:
                lastheard = Event[rainevent][dt_col][-1]

                # Increment the event couter:
                rainevent += 1

            else:
                # if args.verbose:
                #     print("Start rain event %d: %s, last heard %0.3f hrs" % (
                #         rainevent, stormstart, delta_t))

                # Initialize event object arrays:
                # Note, set the rainfall rate boolean here
                

                # Include the starting zero event:
                Event[rainevent][dt_col].append(data[dt_col][i-1])
                Event[rainevent][p_col].append(data[p_col][i-1])

                # Initialize the current rain events rainfall amount:
                eventrain = data[p_col][i]

                # Initialize the iterater:
                j = 0

                while (eventrain > 0):
                    # Save the time and rain amounts for the event:
                    Event[rainevent][dt_col].append(data[dt_col][i+j])
                    Event[rainevent][p_col].append(data[p_col][i+j])

                    # Increment the iterater & update event rainfall:
                    j += 1
                    eventrain = data[p_col][i+j]

                # Include the ending zero event:
                Event[rainevent][dt_col].append(data[dt_col][i+j])
                Event[rainevent][p_col].append(data[p_col][i+j])
                    

                # Update lastheard date:
                lastheard = Event[rainevent][dt_col][-1]

                # Increment the event couter:
                rainevent += 1
    p_events = []
    for i in range(1,len(Event)):
        p_events.append((Event[i][dt_col][0], Event[i][dt_col][-1], np.sum(Event[i][p_col])/((Event[i][dt_col][-1] - Event[i][dt_col][0]).total_seconds()/3600.0), (Event[i][dt_col][-1] - Event[i][dt_col][0]).total_seconds()/3600.0))
    p_events = np.array(p_events)
    p_events = pd.DataFrame(p_events, columns=['start', 'end', 'precip', 'deltat'])
    return p_events

def calc_min_mit(p_events):
    idx, = np.where((np.diff(np.sign(p_events[:,1]-1)) != 0)*1==1)
    if len(idx)==0:
        return 3.0
    return (p_events[:,0][idx][0]+p_events[:,0][idx+1][0])/2.0

def get_events(data, grid_xy):
    gid = '1{gid_x}{gid_y}'.format(gid_x = str(grid_xy[1]).zfill(4), gid_y = str(grid_xy[0]).zfill(4))
    grid_y, grid_x = grid_xy
    p_data = data['p01m'][:,grid_y,grid_x].to_dataframe().reset_index()
    all_data = pd.DataFrame({'dt':data['time'],'p':p_data['p01m']})
    
    if (sum(all_data['p'].isna())/len(all_data)>0.3) | (sum(all_data['p']==0)/len(all_data)>0.99):
        return None

    mit_list = [x for x in range(3,12*24, 6)]
    mit_dry = []
    p_col = 'p'
    dt_col = 'dt'
    for mit in mit_list:
        events = storm_def(all_data, mit, dt_col, p_col)
        dry_periods = [(events['start'][i+1] - events['end'][i]).total_seconds()/3600.0 for i in range(len(events)-1)]
        storm_arrivals = [(events['start'][i+1] - events['start'][i]).total_seconds()/3600.0 for i in range(len(events)-1)]
        mit_dry.append((mit, np.std(dry_periods)/ np.mean(dry_periods),np.std(storm_arrivals)/ np.mean(storm_arrivals), 
        np.mean(dry_periods), np.mean(storm_arrivals)))
    p_events = np.array(mit_dry)
    min_mit = calc_min_mit(p_events)
    events = storm_def(all_data, min_mit, dt_col, p_col)
    return dict({gid: events})


def myfunc(year):
    print(year)
    # year = sys.argv[1]
    ##
    fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/precipitation/stage4/{year}_stage4_hourly.nc'
    out_pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events/{year}.pickle'
    # list of grid_xy tuples: [(grid_x, grid_y), ...,(grid_x, grid_y)] where grid_x varies from 0 to 1120 and grid_y varies from 0 to 880
    perm_list = list(itertools.product(range(881),range(1121)))
    ##
    # year = 2005
    fn_nc_in = fn_fmt.format(year = year)
    data = xar.open_dataset(fn_nc_in)


    result = []
    fn_out_pickle = out_pickle_fmt.format(year=year)
    # n_cpu = cpu_count()
    # pool = ThreadPool(n_cpu)
    for grid_xy in perm_list:
        result.append(get_events(data, grid_xy))
    with open(fn_out_pickle, 'wb') as handle:
        pickle.dump(result, handle, protocol= pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    n_cpu = cpu_count()
    print('# of CPUS working: ' + str(n_cpu))
    pool = ThreadPool(n_cpu)
    permute_list = list([x for x in range(2000,2021)])
    pool.map(myfunc, permute_list)