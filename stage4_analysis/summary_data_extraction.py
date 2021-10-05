import pickle
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import os
import geopandas as gpd

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

pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/monthly/{month}.pickle'


in_pickle = pickle_fmt.format(month = month)
data = read_pickle(in_pickle)
data_all = []
for d in data:
    data_all.extend(d)


events_all = pd.DataFrame(data_all, columns=['grid_xy','tr', 'i', 'tb'])
events_all['h'] = events_all['i'] * events_all['tr']
summary = events_all.groupby('grid_xy').mean().reset_index()
counts = events_all.groupby('grid_xy').count().reset_index()
var_h = events_all.groupby('grid_xy').var().reset_index() # standard deviation of storm depths
mean_h = events_all.groupby('grid_xy').mean().reset_index() # standard deviation of storm depths
summary['lambda'] = mean_h['h']/var_h['h']
summary['kappa'] = mean_h['h']* summary['lambda']
summary['count'] = counts['tr']/21
summary['count'] = summary['count'].astype(np.int32)

summary_path = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/summary'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

fn_summary = os.path.join(summary_path, '{month}.pickle'.format(month = month))
with open(fn_summary, 'wb') as handle:
    pickle.dump(summary, handle, protocol= pickle.HIGHEST_PROTOCOL) 
