import xarray as xr
# import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/precipitation/stage4/{year}_stage4_hourly.nc'
fps = 10
# year = 2004
# instance = (2004,3)
missing_data = [(2003,7),(2003,8), (2004,3)]
for instance in missing_data:
        
    year, month = instance 
    dt_fmt = '{year}-1-1'
    start_dt = dt_fmt.format(year = year)
    end_dt = dt_fmt.format(year = year + 1)

    dt_vec = pd.date_range(start=start_dt, end=end_dt, closed='left', freq='60min')
    dt_vec_m = dt_vec.month

    idx_0 = np.where(dt_vec_m==month)[0][0]
    idx_n = np.where(dt_vec_m==month)[0][-1]
    fn = fn_fmt.format(year = year)
    ds = xr.open_dataset(fn)
    cm1 = plt.cm.get_cmap('Blues',10-0)

    proj=ccrs.LambertConformal(central_longitude=-100)
    fig, ax = plt.subplots(  subplot_kw={'projection': proj},dpi=300)
    air = ds.p01m.isel(time=idx_0)
    # air = air.where(lambda x: x == 0, drop=True)
    # ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
    im = air.plot(x='lon',y='lat', ax=ax,transform=ccrs.PlateCarree(), vmin=1, vmax=10, cmap =cm1)


    # ax.gridlines()
    # plt.savefig('cartopy_example.png', dpi=250)
    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        # ax.text(0.5, 1, str(i) + '\n' + str(dt_vec[idx_0+i]) ,
        #         bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5},
        #         transform=ax.transAxes, ha="center", fontsize=10)
        im.set_array(ds.p01m.isel(time=idx_0+i))
        # ax.set_title('new_title', loc='left')
        ax.set_title(str(dt_vec[idx_0+i]))
        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = idx_n-idx_0,
                                interval = 1000 / fps, # in ms
                                )
    # plt.axis('off')
    ax.coastlines()
    plt.tight_layout()
    anim.save('/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/monthly_totals/trash/{year}_{month}_xarray.mp4'.format(year = year, month = month), fps=fps, bitrate=-1, extra_args=['-vcodec', 'libx264'])