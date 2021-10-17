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
pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/monthly/{month}_2005_2020.pickle'
in_pick_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/{year}_new.pickle'
#
#####
month = int(sys.argv[1])
# month = 1

### Generate year list and shuffle for avoiding I/O problems
year_list = np.arange(2005,2021)
np.random.seed()
np.random.shuffle(year_list)
###

missing_data = [(2004, 3), (2003, 7), (2003, 8)]

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
                if len(data_test[i_y][i_x]) > 2:
                    for m_vec in data_test[i_y][i_x][2:]:
                        if len(m_vec[0]) > 0:
                            bool_month = m_vec[0][0] == month
                            if (year, month) in missing_data:
                                continue
                            if bool_month:
                                event_dur = m_vec[1]
                                dry_vec = m_vec[2]
                                intensity = m_vec[3]
                                n_events = len(event_dur)
                                for i in range(n_events):
                                    _events_all.append([grid_xy, year, month, event_dur[i],intensity[i],dry_vec[i]])
        pickle.dump(_events_all, handle, protocol= pickle.HIGHEST_PROTOCOL) 
