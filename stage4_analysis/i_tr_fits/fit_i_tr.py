import pickle
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




month = int(sys.argv[1])

path_out = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/figures/st4/scaling_i_tr/'
pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/events_mbased/monthly/{month}_new.pickle'
in_pickle = pickle_fmt.format(month = month)
def func1(x, a, b):
    return -a * x + b
def str_mnth(i):
    mnth_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    return mnth_list[i-1]
fit_summary = []
with open(in_pickle, 'rb') as handle:
    try:
        while True:
            data_test = pickle.load(handle)
            events_all = pd.DataFrame(data_test, columns=['grid_xy','year', 'month','tr', 'i', 'tb'])
            if len(events_all)>0:
                # print(events_all['year'].loc[0])
                events_all['h'] = events_all['i'] * events_all['tr']
                year = str(events_all['year'].loc[0])
                fig,ax = plt.subplots(figsize = (15,10))
                for depth in range(10,110,10):
                    x = np.log10(events_all['tr'][((events_all['h']>=depth) & (events_all['h']<depth+10))].values)
                    y = np.log10(events_all['i'][((events_all['h']>=depth) & (events_all['h']<depth+10))].values)

                    popt, pcov = curve_fit(func1, x, y)

                    ax.scatter(x, y,  alpha= 0.01, s = 20 )

                    ax.plot(np.linspace(min(x),max(x),1000), func1(np.linspace(min(x),max(x),1000), *popt), c = 'red')
                    fit_summary.append([depth, popt[0], popt[1], month, year])
                ax.set_xlim([-0.2, 3])
                ax.set_ylim([-2.2, 2.2])
                ax.set_ylabel(r'$i \ (mm/hr)$', fontsize = 24)
                ax.set_xlabel(r'$t_r \ (hr)$', fontsize = 24)

                ax.set_title('{year} - {month}'.format(year = year, month = str_mnth(month)), fontsize = 24)
                ax.tick_params(labelsize = 20)
                fn_out = os.path.join(path_out, '{year}_{month}.jpg'.format(year = year, month = month))
                fig.savefig(fn_out)
    except EOFError:
        pass

summ_out = '/storage/coda1/p-rbras6/0/njadidoleslam3/projects/stochsm/stage4_analysis/i_tr_fits/{month}.pickle'.format(month = month)
with open(summ_out, 'wb') as handle:
    pickle.dump(fit_summary, handle, protocol= pickle.HIGHEST_PROTOCOL) 