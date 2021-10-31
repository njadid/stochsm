import pickle
import numpy as np

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
pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/events_mbased/monthly/{month}.pickle'
in_pick_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/gpm_analysis/events_mbased/yearly/{year}.pickle'

#####
month = int(sys.argv[1])

### Generate year list and shuffle for avoiding I/O problems in multiple job submissions
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
                if len(data_test[i_y][i_x]) > 2:
                    for m_vec in data_test[i_y][i_x][2:]:
                        if len(m_vec[0]) > 0:
                            bool_month = m_vec[0][0] == month
                            if bool_month:
                                event_dur = m_vec[1]
                                dry_vec = m_vec[2]
                                intensity = m_vec[3]
                                n_events = len(event_dur)
                                for i in range(n_events):
                                    _events_all.append([grid_xy, year, month, event_dur[i],intensity[i],dry_vec[i]])
        pickle.dump(_events_all, handle, protocol= pickle.HIGHEST_PROTOCOL) 
