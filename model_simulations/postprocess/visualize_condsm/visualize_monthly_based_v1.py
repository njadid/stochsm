import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib import ticker
import pandas as pd
import geopandas as gpd
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

month = int(str(sys.argv[1]))

###############################################################################
path_out_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/figures/agufihm_v0/maps_v1/{metric}/'
fn_out_fmt = '{metric}_{month}_{stage}_{init_cond}.jpg'
###############################################################################
fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_condsm_monthly_v1/{month}.hdf'
top_layer_depth = 1000

###############################################################################
def smvol2cb(x, theta_r):
    return  x * (top_layer_depth + theta_r * top_layer_depth) - theta_r * top_layer_depth

def myround5(x, base=5):
    return base * np.round(x/base)

def myround25(x, base=0.025):
    return base * np.round(x/base)

def normalize_sm(x, theta_r, theta_s):
    return (x - theta_s) / (theta_s - theta_r)
###############################################################################
# lakes =        gpd.read_file('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/smap_gw/data/gis_data/coastlines_boundariesetc/GSHHS_shp/l/GSHHS_l_L2.shp')
# world =        gpd.read_file('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/smap_gw/data/gis_data/coastlines_boundariesetc/GSHHS_shp/l/GSHHS_l_L1.shp')
# world_states = gpd.read_file('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/smap_gw/data/gis_data/coastlines_boundariesetc/WDBII_shp/l/WDBII_border_l_L1.shp')

s_props = pd.read_csv('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/soil_props/smap_sprops_montzka_30cm_v1.csv')
modeled_grid = gpd.read_file('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/model_simulations/postprocess/modeled_grids/smap_grid_60NS.shp')

fn = fn_fmt.format(month=str(month))
data = pd.read_hdf(fn)
merged = pd.merge(modeled_grid, data, on='grid_xy')
merged_w_sprops  = pd.merge(merged, s_props, on='grid_xy')

merged_w_sprops['init_v0'] = np.around(
    myround25(
        smvol2cb(
            merged_w_sprops['init'], merged_w_sprops['theta_r'])/1000
    ),
    3)


def get_subset(stage, init):
    idx = (merged_w_sprops['stage'] == stage) & (merged_w_sprops['init_v0'] == init)
    return merged_w_sprops[idx]

    

metric_props = {'mean': {'label': r'Mean $[cm^3/cm^3]$', 'min': 0.1, 'max': 0.6, 'cmap': 'rainbow', 'nbins':20},
            'sd': {'label': r'Standard Deviation $[cm^3/cm^3]$', 'min': 0, 'max': 0.1, 'cmap': 'YlOrRd', 'nbins':10}}

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for stage in [1, 2, 3, 4]:
    for init in [0.125, 0.2, 0.35]:
        irrig_grids = get_subset(stage, init=init)
        for metric in ['mean', 'sd']:


            fig = plt.figure(figsize=(20,10))
            ax = plt.axes(projection=ccrs.PlateCarree())

            # irrig_grids = get_subset(stage, init)
            if metric == 'mean':
                cm1 = plt.cm.get_cmap(metric_props[metric]['cmap'], metric_props[metric]['nbins']-0).reversed()
            else:
                cm1 = plt.cm.get_cmap(metric_props[metric]['cmap'], metric_props[metric]['nbins']-0)

            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', linestyle='--', alpha=0.25)
            gl.xlabels_top=False
            gl.ylabels_right=False
            gl.xlines = True
            gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
            gl.ylocator = mticker.FixedLocator([-60, -50, -25, 0, 25, 50, 60])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':16, 'color':'black', 'alpha' : 0.5}
            gl.ylabel_style = {'size':16, 'color':'black', 'alpha' : 0.5}
            
            a = irrig_grids.plot(ax=ax, column = metric,edgecolor = 'none',vmin= metric_props[metric]['min'], vmax = metric_props[metric]['max'], cmap = cm1)

            x_d = np.linspace(metric_props[metric]['min'],\
                    metric_props[metric]['max'],\
                    metric_props[metric]['nbins']+1)

            ax.coastlines(resolution="110m",linewidth=1)
            ax.set_axis_off()
            ax.set_xlim([-180, 180])
            ax.set_ylim([-60, 60])


            cax = fig.add_axes([0.14, 0.15, 0.75, 0.04])
            fig.colorbar( a.collections[0], cax=cax, orientation='horizontal')
            cax.tick_params(labelsize=22)

            cax.set_xlabel(metric_props[metric]['label'], fontsize=22)


            ax.set_title(r'{month}, Stage = +{stage} [days], $\theta_i\ =\ {init}\ [cm^3/cm^3]$'.format(month=months[month-1], stage=str(int(stage*8)), init=str(init)), fontsize=24)

            labels = []
            for i,x1 in enumerate(x_d):
                if i%2==0:

                    labels.append('{:.2f}'.format(x1))
                else:
                    labels.append('')
            cax.xaxis.set_major_locator(ticker.FixedLocator(x_d))
            cax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

            path_out = path_out_fmt.format(metric=metric)
            if os.path.exists(path_out) == False:
                os.makedirs(path_out)

            fn_out = fn_out_fmt.format(month=str(month).zfill(2), stage=str(int(stage*8)).zfill(2), init_cond=str(int(init*1000)).zfill(3), metric=metric)
            plt.savefig(path_out + fn_out, dpi=300, bbox_inches='tight')
            plt.close(fig)