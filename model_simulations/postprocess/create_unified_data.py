import pandas as pd
import numpy as np
import os
import sys
import dask.dataframe as dd
from dask import delayed

def process(fn):
    grid_xy = int(fn.split('/')[-1].split('.')[0])
    # fn_format = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_results_v0/{run_id}/{grid_xy}.csv'
    # fn = fn_format.format(run_id=run_id, grid_xy=grid_xy)
    _data = pd.read_csv(fn)
    _data['grid_xy'] = [grid_xy] * len(_data)
    return _data

run_id = int(str(sys.argv[1]))



if __name__ == '__main__':


    path_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_results_v0/{run_id}/'
    # run_id = 1
    path = path_fmt.format(run_id=run_id)
    if os.path.exists(path):
        _xy_list = os.listdir(path)
        path_list = [os.path.join(path,str(grid_xy)) for grid_xy in _xy_list]


    dfs = [delayed(process)(fname) for fname in path_list]
    ddf = dd.from_delayed(dfs)
    data = ddf.compute()
    data.to_hdf('/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_results_v0_unified/{run_id}.hdf'.format(run_id=run_id), key='data', index=False)