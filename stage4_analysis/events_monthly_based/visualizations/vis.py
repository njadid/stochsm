import os
import sys
import pickle
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from datetime import datetime
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

pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/summary_2005_2020/{month}.pickle'

fn_st_grid_masked = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/data/gis_files/st4/stage4_grid_new.shp'
fn_us_states = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/data/gis_files/conus/conus_states.geojson'

var_list = {
    'tr':       {'max':48, 'min':0, 'label': 'Mean Event Duration [$hr$]', 'bw':6,'nbins':8, 'cmap':'Blues' },
    'i':        {'max':5, 'min':0, 'label': 'Mean Intensity [$mm/hr$]', 'bw':1,'nbins':10,'cmap':'Blues'},
    'tb':       {'max':360, 'min':0, 'label': 'Mean Interarrival Time [$hr$]', 'bw':12, 'nbins':15, 'cmap':'Reds'},
    'h':        {'max':23, 'min':3, 'label': 'Mean Storm Depth [$mm$]', 'bw':3,'nbins':20, 'cmap':'Blues'},
    'count':    {'max':15, 'min':0, 'label': 'Mean Number of Events', 'bw':1,'nbins':15, 'cmap':'Blues'},
    'CV':       {'max':1.2, 'min':0.80, 'label': 'Coef. Var.', 'bw':0.05,'nbins':8, 'cmap':'RdBu'}
    }
# '''
# 'lambda':   {'max':0.26, 'min':0, 'label': 'RMSE [$cm^3/cm^3$]',     'bw':0.01, 'nbins':13, 'cmap':'YlOrRd' },
# 'kappa':    {'max':0.25, 'min':0,  'label': 'MAE [$cm^3/cm^3$]',     'bw':0.01,'nbins':13,'cmap':'Greens'},
# 'cv_tb':    {'max':1.2, 'min':0.8, 'label': '$CV_{tb}$', 'bw':.04,'nbins':10, 'cmap':'RdBu' }}'''




st4_grid_masked = gpd.read_file(fn_st_grid_masked)


us_states = gpd.read_file(fn_us_states)
us_states = us_states.to_crs(4326)

xlim = ([us_states.total_bounds[0],  us_states.total_bounds[2]])
ylim = ([us_states.total_bounds[1],  us_states.total_bounds[3]])




in_pickle = pickle_fmt.format(month = month)
summary = read_pickle(in_pickle)
summary = summary[0]
summary['grid_xy'] = summary['grid_xy'].astype(np.int64)
data = st4_grid_masked.merge(summary)

for variable in list(var_list.keys()):
        


    out_pth = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/figures/st4/summary_mbased_2005_2020/{variable}/'.format(variable = variable)
    out_fn = '{variable}_{month}.jpg'.format(variable = variable, month=month)

    if not os.path.exists(out_pth):
        os.makedirs(out_pth)

    fn_temp = '{variable}.png'
    cm1 = plt.cm.get_cmap(var_list[variable]['cmap'],var_list[variable]['nbins']-0)
    fig, ax = plt.subplots(figsize=(20, 16))

    a = data.plot(ax=ax, column=variable, edgecolor='none', cmap=cm1, vmin = var_list[variable]['min'], vmax=var_list[variable]['max'], zorder = 0)
    us_states.plot(ax=ax, facecolor="none", edgecolor="black", zorder = 1, alpha=0.2)
    
    ax.set_axis_off()
    dt_str = '2020-{month}-1'.format(month = month)
    mnth_name = pd.to_datetime(dt_str).month_name()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.text(0.42,-0.03, mnth_name,transform=ax.transAxes, fontsize=30)
    cax = fig.add_axes([0.1, 0.15, 0.8, 0.04])
    fig.colorbar( a.collections[0], cax=cax, orientation='horizontal')
    cax.set_xlabel(var_list[variable]['label'], fontsize=20)

    cax.tick_params(labelsize=18)
    
    x_d = np.linspace(var_list[variable]['min'], var_list[variable]['max'], var_list[variable]['nbins']+1)
    
    labels = []
    for i,x in enumerate(x_d):
        if i%2==0:
            if variable in ['tb', 'tr', 'count']:
                labels.append(str(int(x)))
            else:
                labels.append(str(x))
        else:
            labels.append('')


    cax.xaxis.set_major_locator(ticker.FixedLocator(x_d))
    cax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # cax.set_xticks(x_d)
    # cax.set_xticklabels([str(x) for x in x_d])
    fn_out = os.path.join(out_pth,out_fn)
    fig.savefig(fn_out, dpi=300, bbox_inches='tight')
    plt.close(fig)