import pickle
import numpy as np
import pandas as pd
import sys
import os

def summarize(data_all):
    events_all = pd.DataFrame(data_all, columns=['grid_xy','year', 'month','tr', 'i', 'tb'])
    events_all['h'] = events_all['i'] * events_all['tr']
    ######### Aggregations #########
    summary = events_all.groupby('grid_xy').mean().reset_index()       # mean
    summary_std = events_all.groupby('grid_xy').std().reset_index()          # standard deviation 
    counts = events_all.groupby('grid_xy').count().reset_index()            # count

    ##########
    summary['CV'] = summary_std['tb']/summary['tb']
    summary['lambda'] = summary['h']/np.power(summary_std['h'],2)
    summary['kappa'] = summary['h'] * summary['lambda']
    summary['count'] = counts['tr']
    return summary

month = int(sys.argv[1])

pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/events/monthly/{month}.pickle'
pickle_out_fmt =  '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/events/summary/{month}.pickle'


in_pickle = pickle_fmt.format(month = month)
fn_out_pickle = pickle_out_fmt.format(month = month)
with open(fn_out_pickle, 'wb') as handle_out:
    with open(in_pickle, 'rb') as handle:
        try:
            while True:
                data_all = pickle.load(handle)
                summary = []
                summary = summarize(data_all)
                pickle.dump(summary, handle_out, protocol= pickle.HIGHEST_PROTOCOL) 
        except EOFError:
            pass