import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from datetime import datetime
import geopandas as gpd
import numpy as np
import h5py
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import glob

def read_pickle(fn):
    data_test = []
    with open(fn, 'rb') as handle:
        try:
            while True:
                data_test.append(pickle.load(handle))
        except EOFError:
            pass
    return data_test


var_list = {
    'tr':       {'max':48, 'min':0, 'label': 'Mean Event Duration [$hr$]', 'bw':6,'nbins':8, 'cmap':'Blues' },
    'i':        {'max':5, 'min':0, 'label': 'Mean Intensity [$mm/hr$]', 'bw':1,'nbins':10,'cmap':'Blues'},
    'tb':       {'max':360, 'min':0, 'label': 'Mean Interarrival Time [$hr$]', 'bw':12, 'nbins':15, 'cmap':'Reds'},
    'h':        {'max':23, 'min':3, 'label': 'Mean Storm Depth [$mm$]', 'bw':3,'nbins':20, 'cmap':'Blues'},
    'count':    {'max':15, 'min':0, 'label': 'Mean Number of Events', 'bw':1,'nbins':15, 'cmap':'Blues'},
    'CV':       {'max':1.2, 'min':0.80, 'label': 'Coef. Var.', 'bw':0.05,'nbins':8, 'cmap':'RdBu'}
    }

# month = int(sys.argv[1])
month = 2

fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/events/summary_mean/{month}.pickle'
fn = fn_fmt.format(month = month)
summary_mean = read_pickle(fn)
summary_mean = summary_mean[0]

data = dict()
for var in list(var_list.keys()):
        data[var] = np.ones((3600,1800))*np.nan

for item in summary_mean.iterrows():    
    grid_y = int(item[1]['grid_xy'] % 10000)
    grid_x = int((item[1]['grid_xy'] - grid_y)/ 10000 % 10000)

    for var in list(var_list.keys()):
        data[var][grid_x, grid_y] = item[1][var]


fn_list = sorted(
    glob.glob(
    '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06/{year}/*/*.HDF5'.format(year=2002)
            )
        )
fn = fn_list[0]
f = h5py.File(fn, 'r')
theLats = f['Grid/lat'][:]
theLons = f['Grid/lon'][:]
x, y = np.float32(np.meshgrid(theLons, theLats))




for variable in list(var_list.keys()):
    out_pth = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/figures/gpm/summary/{variable}/'.format(variable = variable)
    out_fn = '{variable}_{month}.jpg'.format(variable = variable, month=month)

    if not os.path.exists(out_pth):
        os.makedirs(out_pth)

    precip = data[variable]
    precip = np.transpose(precip)

    
    cm1 = plt.cm.get_cmap(var_list[variable]['cmap'],var_list[variable]['nbins']-0)
    # fig, ax = plt.subplots(figsize=(20, 16))

    fig = plt.figure(figsize=(20,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.set_extent([-180,180,-60,60])  

    # Add coastlines and formatted gridlines
    ax.coastlines(resolution="110m",linewidth=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='black', linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([-60, -50, -25, 0, 25, 50, 60])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':16, 'color':'black'}
    gl.ylabel_style = {'size':16, 'color':'black'}

    # Set contour levels and draw the plot
    clevs = np.arange(var_list[variable]['min'], var_list[variable]['max'] + var_list[variable]['max']/var_list[variable]['nbins'],var_list[variable]['max']/var_list[variable]['nbins'])
    plt.contourf(x, y, precip, clevs, cmap=cm1)

    
    ax.set_axis_off()
    dt_str = '2020-{month}-1'.format(month = month)
    mnth_name = pd.to_datetime(dt_str).month_name()
    ax.set_xlim([-180, 180])
    # ax.set_ylim([-60, 60])

    ax.text(0.46,-0.25, mnth_name,transform=ax.transAxes, fontsize=30)
    cax = fig.add_axes([0.15, 0.2, 0.75, 0.04])
    plt.colorbar(ax=ax, orientation='horizontal')
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
    fn_out = os.path.join(out_pth,out_fn)
    fig.savefig(fn_out, dpi=300, bbox_inches='tight')
    plt.close(fig)