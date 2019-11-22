# modules importation
from trends_functions import *


def norm_dist(diff):
    norm = 1
    mu = 0
    stdv = 0.50
    gauss = 100*norm*np.exp(-0.5*(((diff-mu)/stdv)**2))

    consistency = gauss
    return consistency

# - - - - - - run observations - - - - - - - -
# 'ang4487aer' 'od550aer' 'od550gt1aer' 'od550lt1aer' 'sconcpm10' 'sconcpm25' 'sconcso4'
vars = ['od550aer', 'ang4487aer', 'od550gt1aer', 'od550lt1aer', 'concpm10', 'concpm25', 'concso4']
#vars = ['ang4487aer', 'od550gt1aer', 'od550lt1aer', 'sconcpm10', 'sconcpm25', 'sconcso4']
for var in vars:
    print('variable: ',var)

    # computation parameters
    params = get_params()

    params = fill_params(params, var)
    params['var'] = var
    params['kind'] = 'obs'
    obs_source = params['source']
    reader = pya.io.ReadUngridded(obs_source)
    obs_data = reader.read(vars_to_retrieve=var)

    print('OBS')
    OBS_TS, OBS_MAP, OBS_DF = process_trend(
        obs_data, params,
        plot=False, show_plot=False, save_plot=False, write_json=False
    )
    # - - - - - - - - - - - - - - - - - - - - - -


    # - - - - - - -run model - - - - - - -
    #
    mod_var = var
    params['kind'] = 'mod' 
    params['min_dim'] = 0

    mod_source = params['ref_model']
    
    print(mod_source)
    print()

    #check if model in cache
    fn = 'cache/'+mod_source+'_'+var+'.pkl'
    if os.path.isfile(fn):
        print('use pickle')
        # for reading also binary mode is important
        pklfile = open(fn, 'rb')
        mod_data = pickle.load(pklfile)
        pklfile.close()
    else:
        reader = pya.io.ReadGridded(mod_source)
        if (var=='scatc550dryaer'):
            mod_data = reader.read_var(mod_var, ts_type='daily', aux_fun=pya.io.aux_read_cubes.subtract_cubes, aux_vars=['ec550dryaer', 'absc550aer'])
        else:
            mod_data = reader.read_var(mod_var, ts_type='daily')
        #if cube has 4 dimensions, extract first level
        if mod_var in ['concso4', 'concpm10', 'concpm25', 'scatc550dryaer', 'absc550aer'] and len(np.shape(mod_data))==4:
            print('cube has 4 dimension, extract first layer')
            mod_data = mod_data.extract_surface_level()
        mod_data = mod_data.resample_time(to_ts_type='monthly')
        
        #write picke file in cache directory
        pklfile = open(fn, 'ab')

        # source, destination
        pickle.dump(mod_data, pklfile)
        pklfile.close()
    # - - - - - - - - - - - - - - - - - - - - - -

    #crop the cube to interest period, so can handle WORLD region
    mod_data = mod_data.crop(time_range=(params['period'].split('-')[0], str(int(params['period'].split('-')[1])+1)))

    # - - - - - - - - - - - - - - - - - - - - - -
    print('#full colocation')
    MOD_TS, MOD_MAP, MOD_DF = process_trend(
        mod_data, params, obs=obs_data,
        colocate_time=True, colocate_space=True,
        OBS_DF = OBS_DF,
        plot=False, show_plot=False, save_plot=False, write_json=False
    )
    # - - - - - - - - - - - - - - - - - - - - - -



    # - - - - - - - - - - - - - - - - - - - - - -
    #space colocation only
    print('#space colocation only')
    ALLTS_MOD_TS, ALLTS_MOD_MAP, ALLTS_MOD_DF = process_trend(
        mod_data, params, obs=obs_data,
        colocate_time=False, colocate_space=True,
        OBS_DF = OBS_DF,
        plot=False, show_plot=False, save_plot=False, write_json=False,
    )
    # - - - - - - - - - - - - - - - - - - - - - -



    # - - - - - - - - - - - - - - - - - - - - - -
    #all pixels in region
    print('#all pixels in region')
    REG_MOD_TS, REG_MOD_MAP, REG_MOD_DF = process_trend(
        mod_data, params, obs=obs_data,
        colocate_time=False, colocate_space=False,
        OBS_DF = OBS_DF,
        plot=False, show_plot=False, save_plot=False, write_json=False,
    )
    # - - - - - - - - - - - - - - - - - - - - - -





    # - - - - - - -print pickle file- - - - - -
    regions = get_regions()
    MOD = MOD_MAP
    ALLTS_MOD = ALLTS_MOD_MAP
    REG_MOD = REG_MOD_MAP
    key_slope = 'rel_slp'
    kind = 'rel'

    tab = []

    for region in regions:
        #get trends keys
        try:
            periods = MOD[region]['trends']['trends'].keys()
        except KeyError:
            print(region,'Key Error')
            continue
        for p, per in enumerate(periods):
            try:
                r = MOD[region]['trends']['trends'][per]
                allts_r = ALLTS_MOD[region]['trends']['trends'][per]
                reg_r = REG_MOD[region]['trends']['trends'][per]
            except KeyError:
                print('\t',per,'\t','Key Error')
                continue

            str_region = region

            if r[key_slope]!=None and allts_r[key_slope]:
                slp = r[key_slope]
                allts_slp = allts_r[key_slope]
                reg_slp = reg_r[key_slope]

                time_diff = abs(allts_slp - slp)
                space_diff = abs(reg_slp - allts_slp)
                all_diff = np.mean([time_diff, space_diff])

                time_consist = norm_dist(time_diff)
                space_consist = norm_dist(space_diff)
                all_consist = np.mean([time_consist, space_consist])

            else:
                time_diff = np.nan
                space_diff = np.nan
                time_consist = np.nan
                space_consist = np.nan
                all_consist = np.nan
                accuracy = np.nan

            print(params['ylabel'],str_region, per,
                  round(slp,1), round(allts_slp,1), round(reg_slp,1),
                  round(time_diff,1), round(space_diff,1), round(all_diff,1),
                  round(time_consist,1), round(space_consist,1), round(all_consist,1)
                 )
            tab.append([params['ylabel'],str_region, per, all_consist])

    head_df = ['Parameter', 'Region', 'Segment', 'Representativity']
    df = pd.DataFrame(tab, columns=head_df).set_index(['Parameter', 'Region', 'Segment'])

    # write dataframe
    pklfile = open('cache/repr/' + var + '.pkl', 'wb')
    # source, destination
    pickle.dump(df, pklfile)
    pklfile.close()
    # - - - - - - - - - - - - - - - - - - - - - -
