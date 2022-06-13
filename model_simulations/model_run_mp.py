# general libraries import
import pandas as pd
import numpy as np
import preprocess
from calculate_stats import calc_stats
from datetime import datetime
import os
import sys
# import pickle
# import spotpy

# Multiprocessing libraries import
from multiprocessing import Pool
from multiprocessing import cpu_count


import warnings
## End of imports
###
warnings.filterwarnings("ignore")



run_id = int(str(sys.argv[1]))
check_existance = True

#% global variables 
n_nodes = 20
path_out_fmt =      '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_results_v0/{run_id}/'
# path_out_fmt =      '/storage/scratch1/6/njadidoleslam3/gpm_results_v0/{run_id}/'

def start_end_idx(run_id, n_nodes, n_points):
    '''
    Calculates the start and end indices based on number of following parameters:

    Parameters
    ------
    run_id: Int
    Unique ID for the computations deciated to a single Node
    Argument provided from job submission file (e.g., SH, PBS)
    
    n_nodes: Int
    Number of nodes requested from the HPC 

    n_points: Int
    Number of grid points for the calculations

    Return 
    -------
    strat and end indices: int, int
    '''
    chunk_size = int(n_points / n_nodes) 
    start_idx = run_id * chunk_size 
    end_idx = (run_id + 1) * chunk_size
    if run_id == (n_nodes -1):
        end_idx = (run_id +1) * chunk_size + n_points % n_nodes
    return start_idx, end_idx



def compute_probab_sm(grid_xy):
    if os.path.exists(path_out_fmt.format(run_id=run_id) + '{grid_xy}.csv'.format(grid_xy=grid_xy)):
        # print('File exists for grid ' , str(grid_xy))
        return None
    try:
        data_daily = preprocess.preprocess(grid_xy=grid_xy)
    except:
        print('Data retrieval error at grid ' , str(grid_xy))
        return None
    # try:

    # nan_counts = sum(pd.isna(data_daily['gpm_p']))
    # nan_percentage = nan_counts/len(data_daily['gpm_p'])
    
    # if nan_percentage> 0.90:
    #     print('NaN count exceeds at grid ' , str(grid_xy))
    #     return None

    # else:
    try:
        calc_stats(data_daily, s_props, grid_xy, run_id)
    except:
        print('Failed computation at ', str(grid_xy))


#################################################################

if __name__=='__main__':
    # grid_xy_list = pd.read_csv('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/smap_gw/scripts/poster_figures/irrig_points_evren/irrig_points_gridxy.csv')
    grid_xy_list = pd.read_csv('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/smap_grid_list_run.csv')
    # grid_xy_list = pd.read_csv('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/modeled_grids/grid_xy_not_simulated.csv')
    # grid_xy_list = grid_xy_list['grid_xy'][0:5].to_list()
    grid_xy_list = grid_xy_list['grid_xy'].to_list()
    
    n_points = len(grid_xy_list)
    # get start and end indices for the grid_xy points for the run set from job submission
    start_idx, end_idx = start_end_idx(run_id, n_nodes, n_points)
    grid_sublist = grid_xy_list[start_idx:end_idx]

    # Check output path for writing
    main_path_out = path_out_fmt.format(run_id=str(run_id))
    if not os.path.exists(main_path_out):
        os.makedirs(main_path_out)

    s_props = pd.read_csv('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/smap_gw/data/soil_props/smap_sprops_montzka_30cm_v1.csv')

    print('Calculating from ' , str(start_idx), ' to ', str(end_idx))

    # for grid_xy in grid_xy_list:
    #     compute_probab_sm(grid_xy)

    n_cpu = cpu_count()
    pool = Pool(n_cpu)
    t_start = datetime.now()
    metrics_results = pool.map(compute_probab_sm, grid_sublist)   
    # pool.close()
    t_end = datetime.now()
    total_time = (t_end - t_start)/60
    # export_metrics2pickle(metrics_results)
    # export_metrics(metrics_results)
    print('Total Elapsed Time = ' + "%.2f" % total_time.total_seconds() + ' minutes.')
    print('RUN_ID = ', str(run_id), ' Done!')
