# modules importation
from trends_functions import *


# computation parameters
params = {
    # minimum number of daily observations available in order to keep the station
    'min_dobs': 300,
    'min_nstat': 2,  # minimum number of stations required to compute median
    'sig': 0.95,  # significance
    'min_dim': 5,  # minimum number of days required to compute monthly mean
    'min_mis': 1,  # minimum number of months required to compute seasonal mean
    'min_siy': 4,  # minimum number of seasons required to compute annual mean
    'nseg': 2,  # number of segments if no significant linear trend on the time series is found
    # if use same segments for model and bias than the ones found in obs (to be run before)
    'use_obs_seg': True,
    'period': '1995-2018',
    'kind': None,  # 'obs' or 'mod'
    'var': None,
    'source': None,
    'ymin': None,
    'ymax': None,
    'ylabel': None
}

def compute_trend(x, y, params, region):
    # function that provide a linear fit, if significant. If not, break out the function within nseg.

    # mann kendall test
    tau, pval = stats.kendalltau(x, y)

    # theil sen slope
    a, b, low_slope, up_slope = stats.mstats.theilslopes(
        y, x, alpha=params['sig'])
    # reproject on the whole period asked
    xb = np.arange(int(params['period'].split('-')[0]),
                   int(params['period'].split('-')[1]) + 1)
    reg = [a * i + b for i in xb]
    rel_slope = a * 100 / reg[0]

    slope_err = np.mean([abs(a - low_slope), abs(a - up_slope)])
    v0_err_data = np.mean(np.abs(y - [a * i + b for i in x]))
    trend_err = compute_trend_error(m=a, m_err=slope_err, v0=reg[0],
                                    v0_err=v0_err_data
                                    )

    trend = {
        str(xb[0]) + '-' + str(xb[-1]): {
            'x': xb,
            'xdate': [np.datetime64(str(y) + '-06-01') for y in xb],
            'y': reg,
            'a': a,
            'b': b,
            'rel_slp': rel_slope,
            'pval': pval,
            'err_rel_slp': trend_err
        }
    }

    trend1 = copy.copy(trend)
    pval1 = copy.copy(pval)
    short_period = False

    # if (kind=='obs' and pval>(1-sig)) or ((kind!='obs' and use_obs_seg)):
    if (params['kind'] == 'obs') or ((params['kind'] != 'obs' and params['use_obs_seg'])):

        # first, find break points
        #print('needs to break the series')
        # initialize piecewise linear fit with your x and y data
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        if params['kind'] != 'obs' and params['use_obs_seg']:
            #print('use obs segments')
            # fit the data for nseg line segments
            segs = list(OBS_MAP[region]['trends']['trends'].keys())
            if len(segs) <= 1:
                #print('obs do not have segments')
                # print(trend)
                return trend
            else:
                trend = {}
                breaks = [int(segs[0].split('-')[0]),
                          int(segs[0].split('-')[1]), int(segs[1].split('-')[1])]
                #print('obs breaks: ',breaks)
        else:
            trend = {}
            # fit the data for nseg line segments
            breaks = [int(round(x)) for x in my_pwlf.fit(params['nseg'])]

        # loop over segments
        for i in np.arange(params['nseg']):
            mask = [(x >= breaks[i]) & (x <= breaks[i + 1])]
            x2 = x[mask]
            y2 = y[mask]

            if (len(x2) > 3):
                # mann kendall test
                tau, pval = stats.kendalltau(x2, y2)
                # theil sen slope
                a, b, low_slope, up_slope = stats.mstats.theilslopes(
                    y2, x2, alpha=params['sig'])
                # reproject on the whole period asked for first and last segment
                xb = x2
                if i == 0:
                    xb = np.arange(
                        int(params['period'].split('-')[0]), x2[-1] + 1)
                if i == params['nseg']:
                    xb = np.arange(x2[0], int(
                        params['period'].split('-')[1]) + 1)

                reg = [a * i + b for i in xb]
                rel_slope = a * 100 / reg[0]

                slope_err = np.mean([abs(a - low_slope), abs(a - up_slope)])
                v0_err_data = np.mean(np.abs(y2 - [a * i + b for i in x2]))
                trend_err = compute_trend_error(
                    m=a, m_err=slope_err, v0=reg[0], v0_err=v0_err_data)

                trend[str(xb[0]) + '-' + str(xb[-1])] = {
                    'x': xb,
                    'xdate': [np.datetime64(str(y) + '-06-01') for y in xb],
                    'y': reg,
                    'a': a,
                    'b': b,
                    'rel_slp': rel_slope,
                    'pval': pval,
                    'err_rel_slp': trend_err
                }
            else:
                xb = x2
                trend[str(xb[0]) + '-' + str(xb[-1])] = {
                    'x': [],
                    'xdate': [],
                    'y': [],
                    'a': None,
                    'b': None,
                    'rel_slp': None,
                    'pval': None,
                    'err_rel_slp': None
                }
                short_period = True

    # compare p-values
    if params['kind'] == 'obs':
        seg_pval = []
        for seg in trend.keys():
            if (trend[seg]['pval'] != None):
                seg_pval.append(trend[seg]['pval'])
        if min(seg_pval) >= pval1 or short_period == True:
            trend = trend1

    # compute trend along time series
    return trend



def process_trend(data, params, obs=None, colocate_time=True,
                  colocate_space=True, plot=True, write_json=False,
                  show_plot=False, save_plot=False):
    # by default, colocate model in space and time
    MAP, DF, ALL_TS = {}, {}, {}
    regions = pya.region.all()
    regions = ['EUROPE', 'ASIA', 'AUSTRALIA',
               'NAFRICA', 'SAFRICA', 'NAMERICA', 'SAMERICA']

    # computation region by region
    for region in regions:
        f = pya.Filter(region)
        if params['kind'] == 'obs':
            try:
                sub = f(data)
                var = data.vars_to_retrieve[0]
            except:
                print('No station found in the area')
                MAP[region] = {
                    'name': region,
                    'min_lon': pya.Region(region).lon_range[0],
                    'max_lon': pya.Region(region).lon_range[1],
                    'min_lat': pya.Region(region).lat_range[0],
                    'max_lat': pya.Region(region).lat_range[1],
                    'trends': {},
                    'nmax': 0,
                    'stations': [],
                }
                continue
        elif params['kind'] == 'mod':
            if obs == None:
                print('kind is model. Needs to pass an obs dataset')
            else:
                try:
                    sub = f(obs)
                    var = data.var_name
                except:
                    print('No station found in the area')
                    MAP[region] = {
                        'name': region,
                        'min_lon': pya.Region(region).lon_range[0],
                        'max_lon': pya.Region(region).lon_range[1],
                        'min_lat': pya.Region(region).lat_range[0],
                        'max_lat': pya.Region(region).lat_range[1],
                        'trends': {},
                        'nmax': 0,
                        'stations': [],
                    }
                    continue

        # for each subset, creates a dataframe containing all stations timeseries
        # first, get station data
        data_all = sub.to_station_data_all()['stats']
        obs_all = copy.copy(data_all)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # if model
        if params['kind'] == 'mod':
            if colocate_space:
                # first, get list of stations name, lat and lon
                stations = {'name': [], 'lat': [], 'lon': []}

                for stat in data_all:
                    stations['name'].append(stat['station_name'])
                    stations['lat'].append(stat['station_coords']['latitude'])
                    stations['lon'].append(stat['station_coords']['longitude'])

                data_all = data.to_time_series(
                    longitude=stations['lon'], latitude=stations['lat'],
                    add_meta=dict(station_name=stations['name'])
                )
            else:
                print('////////////////////')
                print(type(data))
                #data_all = f(data)
                data_all = f(data)
                med_area = data_all.get_area_weighted_timeseries()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # initialize pandas DataFrame
        df = pd.DataFrame()

        stations = []
        if params['kind'] == 'mod' and colocate_space == False:
            med = med_area[var]
            top = med
            bottom = med
            ts = med.to_frame()
            ts.set_axis([var + '_' + region], axis=1, inplace=True)
            df = pd.concat([df, ts], axis=1)
        else:
            for i, station in enumerate(data_all):
                stat_name = station.station_name
                print('region: ', region, 'station: ', stat_name, end="\r")
                ts_type = station.ts_type
                # set individual time series as dataframe

                # extract pandas series and convert it to datframe
                ts = data_all[i][var].to_frame()
                ts.set_axis([var + '_' + stat_name], axis=1, inplace=True)

                if ts_type == 'daily':
                    if ts.count()[0] >= params['min_dobs']:
                        # concatenates to main dataframe
                        df = pd.concat([df, ts], axis=1)
                else:
                    # concatenates to main dataframe
                    df = pd.concat([df, ts], axis=1)

                stations.append({
                    'name': stat_name,
                    'lat': station.latitude,
                    'lon': station.longitude
                })
            # clear_output(wait=False)

            # caluclates median and envelope with quartiles
            med = df.median(axis=1)
            top = df.quantile(q=0.75, axis=1)
            bottom = df.quantile(q=0.25, axis=1)
            # requires at least n measurements to provide a valid median
            n = df.count(axis=1, numeric_only=False)
            med = med[n > params['min_nstat']]
            top = top[n > params['min_nstat']]
            bottom = bottom[n > params['min_nstat']]

        # if model, colocates in time with obs
        if params['kind'] == 'mod' and colocate_time:
            if region in OBS_DF.keys():
                df = df[OBS_DF[region].shift(-12, freq='H') >= 0]
                # caluclates median and envelope with quartiles
                med = df.median(axis=1)
                top = df.quantile(q=0.75, axis=1)
                bottom = df.quantile(q=0.25, axis=1)
                # requires at least n measurements to provide a valid median
                n = df.count(axis=1, numeric_only=False)
                med = med[n > params['min_nstat']]
                top = top[n > params['min_nstat']]
                bottom = bottom[n > params['min_nstat']]
            else:
                continue

        # calculates monthly averages for the plots
        dcount = med.groupby(pd.Grouper(freq='M')).count()
        mmed = med.groupby(pd.Grouper(freq='M')).mean().where(
            dcount >= params['min_dim'])
        mtop = top.groupby(pd.Grouper(freq='M')).mean().where(
            dcount >= params['min_dim'])
        mbottom = bottom.groupby(pd.Grouper(
            freq='M')).mean().where(dcount >= params['min_dim'])

        # seasonal averages
        mcount = mmed.groupby(pd.Grouper(freq='Q')).count()
        smed = mmed.groupby(pd.Grouper(freq='Q')
                            ).mean().where(mcount >= params['min_mis'])

        # yearly averages from seasonal averages
        scount = smed.groupby(pd.Grouper(freq='A')).count()
        ymed = smed.groupby(pd.Grouper(freq='A')
                            ).mean().where(scount >= params['min_siy'])
        ymed = ymed.shift(-6, freq='MS')

        y_min = int(params['period'].split('-')[0])
        y_max = int(params['period'].split('-')[1])

        # prepare arrays for trends computation
        x = ymed.index.year.values
        xplot = ymed.index
        y = ymed.values
        # get only valid values
        x = x[~np.isnan(y)]
        xplot = xplot[~np.isnan(y)]
        y = y[~np.isnan(y)]
        # restrict to years within period
        xok, yok = [], []
        for i, _ in enumerate(x):
            if x[i] >= y_min and x[i] <= y_max:
                xok.append(x[i])
                yok.append(y[i])
        xok = np.array(xok)
        yok = np.array(yok)

        # write ts json file
        TS = {
            "daily": {
                "jsdate": to_jsdate(med.index).tolist(),
                "data": med.values.tolist(),
                "top": top.values.tolist(),
                "bottom": bottom.values.tolist()
            },
            "monthly": {
                "jsdate": to_jsdate(mmed.index).tolist(),
                "data": mmed.values.tolist(),
                "top": mtop.values.tolist(),
                "bottom": mbottom.values.tolist()
            },
            "yearly": {
                "jsdate": to_jsdate(ymed.index).tolist(),
                "data": ymed.values.tolist(),
            },
            "trends": {}
        }

        # trends computation
        if len(xok) == 0:
            print('No Data Available in Selected Period')
            MAP[region] = {
                'name': region,
                'min_lon': pya.Region(region).lon_range[0],
                'max_lon': pya.Region(region).lon_range[1],
                'min_lat': pya.Region(region).lat_range[0],
                'max_lat': pya.Region(region).lat_range[1],
                'trends': TS,
                'nmax': len(df.columns),
                'stations': stations
            }
            continue
        trend = compute_trend(xok, yok, params, region)

        if plot:
            plotTS(mmed, mbottom, mtop, ymed, trend,
                   region, params, show_plot, save_plot)

        # store every portions of the trend
        for seg in trend.keys():
            if trend[seg]['rel_slp'] != None:
                jsdate = to_jsdate(trend[seg]['xdate']).tolist()
            else:
                jsdate = []
            TS["trends"][seg] = {
                'pval': trend[seg]['pval'],
                'rel_slp': trend[seg]['rel_slp'],
                'abs_slp': trend[seg]['a'],
                'reg0': trend[seg]['b'],
                'n': len(trend[seg]['x']),
                'data': trend[seg]['y'],
                'jsdate': jsdate,
                'err_rel_slp': trend[seg]['err_rel_slp']
            }

        # export TS as json file
        if write_json:
            write_ts(TS, region, params)

        # append to map dict
        MAP[region] = {
            'name': region,
            'min_lon': pya.Region(region).lon_range[0],
            'max_lon': pya.Region(region).lon_range[1],
            'min_lat': pya.Region(region).lat_range[0],
            'max_lat': pya.Region(region).lat_range[1],
            'trends': TS,
            'nmax': len(df.columns),
            'stations': stations,
        }

        ALL_TS[region] = TS
        DF[region] = df

    # export MAP as json file
    if write_json:
        write_map(MAP, params)
        write_all_ts(ALL_TS, params)

    return TS, MAP, DF


# run observations
# 'ang4487aer' 'od550aer' 'od550gt1aer' 'od550lt1aer' 'sconcpm10' 'sconcpm25' 'sconcso4'
var = 'od550aer'
params['kind'] = 'obs'
if var == 'od550aer':
    params['source'] = 'AeronetSunV3Lev2.daily'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD'
    params['min_dim'] = 5
if var == 'ang4487aer':
    params['source'] = 'AeronetSunV3Lev2.daily'
    params['ymin'] = 0
    params['ymax'] = 2.5
    params['ylabel'] = 'AE'
    params['min_dim'] = 5
if var == 'od550gt1aer':
    params['source'] = 'AeronetSDAV3Lev2.daily'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD>1µm'
    params['min_dim'] = 5
if var == 'od550lt1aer':
    params['source'] = 'AeronetSDAV3Lev2.daily'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD<1µm'
    params['min_dim'] = 5
if var == 'sconcpm10':
    params['source'] = 'EBASMC'
    params['ymin'] = 0
    params['ymax'] = 60
    params['ylabel'] = 'PM10'
    params['min_dim'] = 4  # weekly measurements
if var == 'sconcpm25':
    params['source'] = 'EBASMC'
    params['ymin'] = 0
    params['ymax'] = 30
    params['ylabel'] = 'PM2.5'
    params['min_dim'] = 4  # weekly measurements
if var == 'sconcso4':
    params['source'] = 'EBASMC'
    params['ymin'] = 0
    params['ymax'] = 6
    params['ylabel'] = 'SO4'
    params['min_dim'] = 4  # weekly measurements

params['var'] = var
obs_source = params['source']
reader = pya.io.ReadUngridded(obs_source)
obs_data = reader.read(vars_to_retrieve=var)

print('OBS')
OBS_TS, OBS_MAP, OBS_DF = process_trend(
    obs_data, params,
    plot=True, show_plot=False, save_plot=True, write_json=True
)

# run model
mod_var = var
params['kind'] = 'mod'
if var == 'od550aer':
    params['source'] = 'ECMWF_CAMS_REAN'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD'
    params['min_dim'] = 0
if var == 'ang4487aer':
    params['source'] = 'ECMWF_CAMS_REAN'
    params['ymin'] = 0
    params['ymax'] = 2.5
    params['ylabel'] = 'AE'
    params['min_dim'] = 0
if var == 'od550gt1aer':
    params['source'] = 'GFDL-AM4-amip_HIST'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD>1µm'
    params['min_dim'] = 0
if var == 'od550lt1aer':
    params['source'] = 'GFDL-AM4-amip_HIST'
    params['ymin'] = 0
    params['ymax'] = 0.8
    params['ylabel'] = 'AOD<1µm'
    params['min_dim'] = 0
if var == 'sconcpm10':
    params['source'] = 'ECMWF_CAMS_REAN'
    params['ymin'] = 0
    params['ymax'] = 60
    params['ylabel'] = 'PM10'
    params['min_dim'] = 4  # weekly measurements
if var == 'sconcpm25':
    params['source'] = 'ECMWF_CAMS_REAN'
    params['ymin'] = 0
    params['ymax'] = 30
    params['ylabel'] = 'PM2.5'
    params['min_dim'] = 0
if var == 'sconcso4':
    params['source'] = 'ECMWF_CAMS_REAN'
    params['ymin'] = 0
    params['ymax'] = 6
    params['ylabel'] = 'SO4'
    params['min_dim'] = 0

mod_source = params['source']
if 'mod_data' in locals():
    print('model alread in memory')
else:
    print('model reading')
    reader = pya.io.ReadGridded(mod_source)
    mod_data = reader.read_var(mod_var, ts_type='daily')
    mod_data = mod_data.resample_time(to_ts_type='monthly')


#full colocation
print('#full colocation')
MOD_TS, MOD_MAP, MOD_DF = process_trend(
    mod_data, params, obs=obs_data,
    colocate_time=True, colocate_space=True,
    plot=False, show_plot=False, save_plot=False, write_json=False
)

#space colocation only
print('#space colocation only')
ALLTS_MOD_TS, ALLTS_MOD_MAP, ALLTS_MOD_DF = process_trend(
    mod_data, params, obs=obs_data, 
    colocate_time=False, colocate_space=True, 
    plot=False, show_plot=False, save_plot=False, write_json=False, 
)


#all pixels in region
print('#all pixels in region')
REG_MOD_TS, REG_MOD_MAP, REG_MOD_DF = process_trend(
    mod_data, params, obs=obs_data, 
    colocate_time=False, colocate_space=False, 
    plot=False, show_plot=False, save_plot=False, write_json=False, 
)


#print the results
print(' * * OBS * *')
print_trends(OBS_MAP)
print(' * * MOD * *')
print_trends(MOD_MAP)
print(' * * ALLTS_MOD * *')
print_trends(ALLTS_MOD_MAP)
print(' * * REG_MOD * *')
print_trends(REG_MOD_MAP)

#plot the scatter plot
scat_trends(
    OBS_MAP, MOD_MAP, params, obs_source, mod_source, 
    show_plot=True, save_plot=True
)

#computation region by region
print(' * * CONSISTENCY * *')
print_consistency(MOD_MAP, ALLTS_MOD_MAP, REG_MOD_MAP, kind='rel')
