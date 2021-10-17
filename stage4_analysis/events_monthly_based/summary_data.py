import pickle
import numpy as np
import pandas as pd
import sys
# import itertools
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from datetime import datetime
import os
# import geopandas as gpd

def read_pickle(fn):
    data_test = []
    with open(fn, 'rb') as handle:
        try:
            while True:
                data_test.append(pickle.load(handle))
        except EOFError:
            pass
    return data_test



month = int(sys.argv[1])


pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/monthly/{month}_2005_2020.pickle'

summary_path = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/summary_2005_2020'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)



in_pickle = pickle_fmt.format(month = month)
data = read_pickle(in_pickle)
data_all = []
for d in data:
    data_all.extend(d)


events_all = pd.DataFrame(data_all, columns=['grid_xy','year', 'month','tr', 'i', 'tb'])
events_all['h'] = events_all['i'] * events_all['tr']

### Filter the events that are less than 3 mm depth
# events_all = events_all[events_all['h'].values > 3]

######### Aggregations #########
summary = events_all.groupby('grid_xy').mean().reset_index()       # mean
summary_std = events_all.groupby('grid_xy').std().reset_index()          # standard deviation 
counts = events_all.groupby('grid_xy').count().reset_index()            # count



##########
summary['CV'] = summary_std['tb']/summary['tb']
summary['lambda'] = summary['h']/np.power(summary_std['h'],2)
summary['kappa'] = summary['h'] * summary['lambda']
summary['count'] = counts['tr']/16



fn_summary = os.path.join(summary_path, '{month}.pickle'.format(month = month))
with open(fn_summary, 'wb') as handle:
    pickle.dump(summary, handle, protocol= pickle.HIGHEST_PROTOCOL) 
