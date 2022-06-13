import pickle
import numpy as np
import pandas as pd
import sys
import os




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

pickle_out_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/events/summary_mean/{month}.pickle'
fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/events/summary/{month}.pickle'
fn = fn_fmt.format(month=month)
data = read_pickle(fn)
columns = {'grid_xy': np.int32,	'year': np.int32,	'month': np.int32,	'tr': np.float32,	'i': np.float32,
           'tb': np.float32,	'h': np.float32,	'CV': np.float32,	'lambda': np.float32,	'kappa': np.float32,	'count': np.float32}
summary = pd.DataFrame(columns=columns)
summary.astype(columns)
for item in data:
    summary = summary.append(item, ignore_index=True)

summary = summary.astype(columns)
summary_mean = summary.groupby('grid_xy').mean().reset_index()

fn_out_pickle = pickle_out_fmt.format(month=month)
with open(fn_out_pickle, 'wb') as handle_out:
    pickle.dump(summary_mean, handle_out, protocol=pickle.HIGHEST_PROTOCOL)
