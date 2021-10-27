import numpy as np
import pickle
import itertools
import h5py
import glob
import timeit
#######################

def extract_events(year):
    def storm_def_new(p_data, mit):
        v = (p_data==0)*1
        n = v==0
        a = ~n
        c = np.cumsum(a)
        d = np.diff(np.append([0.], c[n]))
        v[n] = -d
        dry_vec = np.cumsum(v)
        # mit = 6
        aa = np.append([1], np.diff(dry_vec))
        idx = (aa  <=-mit)*(aa<0)
        dry_periods = aa[idx]*-1
        return np.array([mit, np.std(dry_periods[1:]) / np.mean(dry_periods[1:]), np.mean(dry_periods[1:])])

    #######################

    def concatenate(year):
        fn_list_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/time/{year}/*.h5'
        file_names = sorted(glob.glob(fn_list_fmt.format(year = year)))
        temp_h5_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/gpm/virt_ds/{year}.h5'
        entry_key = 'precipitationCal'  # where the data is inside of the source files.
    
        fn_1 = file_names[0]
        out_fn = temp_h5_fmt.format(year=year)
        if os.path.exists(out_fn):
            return out_fn

        sh = h5py.File(fn_1, 'r')[entry_key].shape  # get the first ones shape.
        layout = h5py.VirtualLayout(shape=(len(file_names),) + sh, dtype=float)
        with h5py.File(out_fn, 'w', libver='latest') as f:
            for i, filename in enumerate(file_names):
                vsource = h5py.VirtualSource(filename, entry_key, shape=sh)
                layout[i, :, :, :] = vsource

            f.create_virtual_dataset(entry_key, layout, fillvalue=-9999.9)
        return out_fn

    #######################

    # @jit('(f4[:],int32)')
    def calc_min_mit(p_events):
        idx, = np.where((np.diff(np.sign(p_events[:,1]-1)) != 0)*1==1)
        if len(idx)==0:
            a = np.abs(p_events[:,1]-1)
            idx, = np.where(a == a.min())
            return p_events[:,0][idx][0]
        return (p_events[:,0][idx][0]+p_events[:,0][idx+1][0])/2.0

    #######################

    def get_events(p_data,set_min_mit):
        mit_list = [x for x in range(set_min_mit,12*24, 6)]
        mit_dry = []
        for mit in mit_list:
            mit_dry.append(storm_def_new(p_data, mit))
            
        mit_dry = np.array(mit_dry)
        min_mit = calc_min_mit(mit_dry)
        event_metrics = storm_def_new(p_data, min_mit)
        # print(event_metrics)
        return event_metrics
    
    set_min_mit = 1
    fn_in = concatenate(year)
    # %timeit f = h5py.File(fn_in,'r') 
    f = h5py.File(fn_in,'r') 
    # %timeit dset = f['precipitationCal']
    dset = f['precipitationCal']

    out_pickle_fmt = '/storage/coda1/p-rbras6/0/njadidoleslam3/precipitation/gpm/events/{year}.pickle'
    fn_out_pickle = out_pickle_fmt.format(year=year)

    ####
    ## NOTE Grid_x and Grid_y in GPM
    ## for STAGE 4 it is grid_y and grid_x
    # list of grid_xy tuples: [(grid_x, grid_y), ...,(grid_x, grid_y)] where grid_x varies from 0 to 1120 and grid_y varies from 0 to 880
    ##########
    # perm_list = np.array([itertools.product(np.arange(3600),np.arange(1800))])[500000:500100]##
    ####

    perm_list = [(500,1000), (500, 1001)] ## Comment this line for full run
    
    result = np.empty((1,1), dtype=object)
    for grid_xy in perm_list:
        grid_x, grid_y = grid_xy
        gid = int('1{gid_x}{gid_y}'.format(gid_x = str(grid_x).zfill(4), gid_y = str(grid_y).zfill(4)))
        
        p_data = np.ravel(dset[:, :, grid_x, grid_y])
        # %timeit p_data1 =  p_data.reshape((int(p_data.shape[0]/2),2)).mean(axis=1)
        p_data =  p_data.reshape((int(p_data.shape[0]/2),2)).mean(axis=1)
        n_data = len(p_data)
        # event_metrics = get_events(p_data, set_min_mit)
        # %timeit np.sum(p_data < 0)
        pos_idx = np.sum(p_data>0)
        neg_idx = ~pos_idx
        if float(np.sum(neg_idx)/n_data)<0.1:
            # print('No data 1', float(np.sum(p_data < 0)/n_data))
            p_data[p_data<0] = 0
            
        if (float(np.sum(pos_idx)/n_data)<0.10):
            result = np.append(result,np.array([gid,np.nan,np.nan,np.nan]))
            # print('No data 2')
        else:
            try:
                event_metrics = get_events(p_data, set_min_mit)
                result = np.append(result,np.concatenate((gid, event_metrics),axis=None))
            except:
                result = np.append(result,np.array([gid,np.nan,np.nan,np.nan]))

    
    with open(fn_out_pickle, 'wb') as handle:
        pickle.dump(result, handle, protocol= pickle.HIGHEST_PROTOCOL)
    return result

year = sys.argv[1]
result = extract_events(year)
