fn_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/gpm_condsm_monthly_v1/{month}.hdf'
top_layer_depth = 1000
def smvol2cb(x, theta_r):
    return  x * (top_layer_depth + theta_r * top_layer_depth) - theta_r * top_layer_depth

def myround(x, base=5):
    return base * np.round(x/base)
    
def myround25(x, base=0.025):
    return base * np.round(x/base)

def myround5(x, base=0.025):
    return base * np.round(x/base)
    
def normalize_sm(x, theta_r, theta_s):
    return (x - theta_r) / (theta_s - theta_r)

fn = fn_fmt.format(month='1')
data = pd.read_hdf(fn)
merged = pd.merge(modeled_grid, data, on='grid_xy')
merged_w_sprops  = pd.merge(merged, s_props, on='grid_xy')
init_v0 = np.around(
    myround25(
        smvol2cb(
            merged_w_sprops['init'], merged_w_sprops['theta_r'])/1000
    ),
    3)
# norm_init_sm = np.around(myround25(normalize_sm(init_v0, merged_w_sprops['theta_r'], merged_w_sprops['theta_s'])), 3)
# norm_init_sm = np.around(myround25(normalize_sm(init_v0, merged_w_sprops['theta_r'], merged_w_sprops['theta_s'])), 3)
merged_w_sprops['mean'] = smvol2cb(merged_w_sprops['mean'],merged_w_sprops['theta_r'])/merged_w_sprops['theta_s']/1000
merged_w_sprops['sd'] = merged_w_sprops['sd']/(merged_w_sprops['theta_s']- merged_w_sprops['theta_r'])
def myround75(x, base=0.075):
    return base * np.round(x/base)
norm_init_sm = np.around(myround75(normalize_sm(init_v0, merged_w_sprops['theta_r'], merged_w_sprops['theta_s'])), 3)

merged_w_sprops['init_v0'] = norm_init_sm


