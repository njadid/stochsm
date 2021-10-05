import pickle
import numpy as np
import pandas as pd
import os
import sys
# Reorganize data by month
def read_pickle(fn):
    data_test = []
    with open(fn, 'rb') as handle:
        try:
            while True:
                data_test.append(pickle.load(handle))
        except EOFError:
            pass
    return data_test


#####
# SHAPE FILES  TO BE ADDED TO GPM ALGORITHM
# fn_st_grid_masked = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/data/gis_files/st4/stage4_grid_new.shp'
# st4_grid_masked = gpd.read_file(fn_st_grid_masked)
#####
pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/monthly/{month}_new.pickle'
in_pick_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events/{year}_new.pickle'
#
#####
month = int(sys.argv[1])
# month = 1

### Generate year list and shuffle for avoiding I/O problems
year_list = np.arange(2000,2021)
np.random.seed()
np.random.shuffle(year_list)
###

fn_out_pickle = pickle_fmt.format(month = month)

with open(fn_out_pickle, 'wb') as handle:
# events_all = []
    for year in year_list:
        in_pickle = in_pick_fmt.format(year = year)
        data_test = read_pickle(in_pickle)

        _events_all = []
        for i_y in range(len(data_test)): # loop over grid_y
            for i_x in range(len(data_test[i_y])): # loop over grid_x
                grid_xy = int(data_test[i_y][i_x][0])
                idx = data_test[i_y][i_x][3][0] == month
                n_events = sum(idx)
                if n_events >0:
                    event_dur = data_test[i_y][i_x][3][1][idx]
                    dry_vec = data_test[i_y][i_x][3][2][idx]
                    intensity = data_test[i_y][i_x][3][3][idx]
                    for i in range(n_events):
                        _events_all.append([grid_xy, event_dur[i],intensity[i],dry_vec[i]])
    # with open(fn_out_pickle, 'wb') as handle:
        pickle.dump(_events_all, handle, protocol= pickle.HIGHEST_PROTOCOL) 


##### Summarize the data and save the dataframe to 
# events_all = pd.DataFrame(events_all, columns=['grid_xy','tr', 'i', 'tb'])
# events_all['h'] = events_all['i'] * events_all['tr']
# summary = events_all.groupby('grid_xy').mean().reset_index()
# counts = events_all.groupby('grid_xy').count().reset_index()
# var_h = events_all.groupby('grid_xy').var().reset_index() # standard deviation of storm depths
# mean_h = events_all.groupby('grid_xy').mean().reset_index() # standard deviation of storm depths
# summary['lambda'] = mean_h['h']/var_h['h']
# summary['kappa'] = mean_h['h']* summary['lambda']
# summary['count'] = counts['tr']/21

# #### SAVE summary data to pickle files 
# summary_path = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/summary'
# if not os.path.exists(summary_path):
#     os.makedirs(summary_path)

# fn_summary = os.path.join(summary_path, '{month}.pickle'.format(month = month))
# with open(fn_summary, 'wb') as handle:
#     pickle.dump(summary, handle, protocol= pickle.HIGHEST_PROTOCOL) 


## END