# dedicated functions for trends computation
import pyaerocom as pya
import pandas as pd
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import scipy.stats as stats
import simplejson as json
import numpy as np
import datetime
import copy
import sys
import pwlf
import seaborn as sns
import pickle
import os

path_out = '../../aerosoltrends/data/test/'
#pya.change_verbosity('error')
pya.change_verbosity('critical', pya.const.logger)
pya.change_verbosity('critical', pya.const.print_log)

def get_params():
    # computation parameters
    params = {
        'min_dobs': 300, # minimum number of daily observations available in order to keep the station
        'min_ntrend': 7, #minimum number of points used  to compute a trend
        'min_nstat': 2,  # minimum number of stations required to compute median
        'sig': 0.95,  # significance
        'min_dim': 5,  # minimum number of days required to compute monthly mean
        'min_mis': 1,  # minimum number of months required to compute seasonal mean
        'min_siy': 4,  # minimum number of seasons required to compute annual mean
        'nseg': 2,  # number of segments if no significant linear trend on the time series is found
        # if use same segments for model and bias than the ones found in obs (to be run before)
        'use_obs_seg': True,
        'period': '2000-2014',
        'kind': None,  # 'obs' or 'mod'
        'var': None,
        'source': None,
        'ymin': None,
        'ymax': None,
        'ylabel': None
    }
    return params


def fill_params(params, var):
    #by default, same variable is used in models
    params['mod_var'] = var
    #for the models, redce min_ntrend to 4, so still
    if var == 'od550aer':
        params['source'] = 'AeronetSunV3Lev2.daily'
        params['ymin'] = 0
        params['ymax'] = 0.8
        params['ylabel'] = 'AOD'
        params['min_dim'] = 5
        params['models'] = ['ECMWF_CAMS_REAN', 
                            'OsloCTM3v1.01-met2010_AP3-HIST', 'NorESM2-LM_historical', 
                 'BCC-CUACE_HIST', 'CAM5-ATRAS_AP3-HIST', 'GFDL-AM4-amip_HIST', 'CanESM5_historical', 
                 'CESM2_historical', 'IPSL-CM6A-LR_historical', 'GEOS-i33p2_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']
        params['ref_model'] = 'GFDL-AM4-amip_HIST'
    if var == 'ang4487aer':
        params['source'] = 'AeronetSunV3Lev2.daily'
        params['ymin'] = 0
        params['ymax'] = 2.5
        params['ylabel'] = 'AE'
        params['min_dim'] = 5
        params['models'] = ['ECMWF_CAMS_REAN', 'OsloCTM3v1.01-met2010_AP3-HIST', 
                   'CAM5-ATRAS_AP3-HIST', 'GFDL-AM4-amip_HIST', 'GEOS-i33p2_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']
        params['ref_model'] = 'GFDL-AM4-amip_HIST'
    if var == 'od550gt1aer':
        params['source'] = 'AeronetSDAV3Lev2.daily'
        params['ymin'] = 0
        params['ymax'] = 0.8
        params['ylabel'] = 'AOD>1µm'
        params['min_dim'] = 5
        params['models'] = ['ECMWF_CAMS_REAN', 'OsloCTM3v1.01-met2010_AP3-HIST', 
                    'BCC-CUACE_HIST', 'CAM5-ATRAS_AP3-HIST', 'GFDL-AM4-amip_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']
        params['ref_model'] = 'GFDL-AM4-amip_HIST'
    if var == 'od550lt1aer':
        params['source'] = 'AeronetSDAV3Lev2.daily'
        params['ymin'] = 0
        params['ymax'] = 0.8
        params['ylabel'] = 'AOD<1µm'
        params['min_dim'] = 5
        #params['models'] = ['OsloCTM3v1.01-met2010_AP3-HIST', 'CAM5-ATRAS_AP3-HIST', 'GFDL-AM4-amip_HIST']
        params['models'] = ['GFDL-AM4-amip_HIST', 'GEOS-i33p2_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']
        params['ref_model'] = 'GFDL-AM4-amip_HIST'
    if var == 'concpm10':
        params['source'] = 'EBASMC'
        params['ymin'] = 0
        params['ymax'] = 60
        params['ylabel'] = 'PM10'
        params['min_dim'] = 4  # weekly measurements
        params['models'] = ['ECMWF_CAMS_REAN', 'GEOS-i33p2_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']
        params['ref_model'] = 'ECMWF_CAMS_REAN'
    if var == 'concpm25':
        params['source'] = 'EBASMC'
        params['ymin'] = 0
        params['ymax'] = 30
        params['ylabel'] = 'PM2.5'
        params['min_dim'] = 4  # weekly measurements
        params['models'] = ['ECMWF_CAMS_REAN']
        params['ref_model'] = 'ECMWF_CAMS_REAN'
    if var == 'concso4':
        params['source'] = 'GAWTADsubsetAasEtAl'
        params['ymin'] = 0
        params['ymax'] = 6
        params['ylabel'] = 'SO4'
        params['min_dim'] = 0  # monthly measurements
        params['models'] = ['GEOS-i33p2_HIST', 'ECHAM6.3-HAM2.3-fSST_HIST']   
        params['ref_model'] = 'GEOS-i33p2_HIST'
    if var == 'scatc550dryaer':
        params['source'] = 'EBASMC'
        params['ymin'] = 0
        params['ymax'] = 100
        params['ylabel'] = 'Scat. Coef.'
        params['min_dim'] = 5
        params['min_nstat'] = 2
        params['models'] = []
        params['ref_model'] = None
    if var == 'absc550aer':
        params['source'] = 'EBASMC'
        params['ymin'] = 0
        params['ymax'] = 10
        params['ylabel'] = 'Abs. Coef.'
        params['min_dim'] = 5
        params['min_nstat'] = 2
        params['models'] = []
        params['ref_model'] = None
    return params

def get_all_mods():
    all_mods = {
        'ECMWF_CAMS_REAN': {
            'group': 'group0'
        },
        'GEOS-i33p2_HIST': {
            'group': 'group1'
        },
        'OsloCTM3v1.01-met2010_AP3-HIST': {
            'group': 'group1'
        },
        'NorESM2-LM_historical': {
            'group': 'group1'
        },
        'CAM5-ATRAS_AP3-HIST': {
            'group': 'group1'
        },
        'GFDL-AM4-amip_HIST': {
            'group': 'group1'
        },
        'CanESM5_historical': {
            'group': 'group1'
        },
        'CESM2_historical': {
            'group': 'group1'
        },
        'IPSL-CM6A-LR_historical': {
            'group': 'group1'
        },
        'ECHAM6.3-HAM2.3-fSST_HIST': {
            'group': 'group1'
        }
    }
    #'MIROC-SPRINTARS' #E3SM-1-0_historical #BCC-CUACE_HIST
    return all_mods



def get_color_mod(mod):
    #prepare colors
    current_palette = sns.color_palette("deep", 18)
    colors = {}
    for i, m in enumerate(list(get_all_mods().keys())):
        colors[m] = current_palette[i]
    colors['OBS'] = 'black'
    return colors[mod]

def get_regions():
    regions = ['EUROPE', 'ASIA', 'NAMERICA', 'SAMERICA', 'NAFRICA', 'SAFRICA', 'AUSTRALIA', 'WORLD']
    return regions

def to_jsdate(dates):
    """Convert datetime vector to jsdate vector"""
    epoch = np.datetime64('1970-01-01')
    return (dates - epoch).astype('timedelta64[ms]').astype(int)

def js2date(dates):
    """Convert jsdate vector to datetime vector"""
    secs = [date / 1000 for date in dates]
    dts = [datetime.datetime.fromtimestamp(sec) for sec in secs]
    return dts

def write_ts(TS, region, params):
    source = 'AERONET-Sun'
    layer = 'Column'
    fn = path_out + 'regions_ts/OBS-' + source + ':' + params['var'] + \
        '_' + layer + '_MOD-None:None_' + region + '.json'
    # export TS as json file
    with open(fn, 'w') as fp:
        json.dump(TS, fp, sort_keys=True, ignore_nan=True)

def write_all_ts(TS, params):
    source = 'AERONET-Sun'
    layer = 'Column'
    fn = path_out + 'regions_all_ts/OBS-' + source + ':' + params['var'] + \
        '_' + layer + '_MOD-None:None' + '.json'
    # export TS as json file
    with open(fn, 'w') as fp:
        json.dump(TS, fp, sort_keys=True, ignore_nan=True)


def write_map(MAP, params):
    source = 'AERONET-Sun'
    layer = 'Column'
    fn = path_out + 'regions_map/OBS-' + source + ':' + params['var'] + \
        '_' + layer + '_MOD-None:None' + '.json'
    json.dumps(MAP, sort_keys=True, default=str)
    with open(fn, 'w') as fp:
        json.dump(MAP, fp, sort_keys=True, ignore_nan=True)


def print_trends(MAP):
    bold_start = '\033[1m'
    bold_end = '\033[0m'
    italic_start = '('  # '\x1B[3m'
    italic_end = ')'  # '\x1B[23m'
    print('region', '\t', 'period', '\t', '%/yr', '\t\t', 'pval')
    print(' - - -', '\t', ' - - -', '\t', ' - -', '\t\t', ' - -')
    # computation region by region
    regions = ['EUROPE', 'ASIA', 'AUSTRALIA',
               'NAFRICA', 'SAFRICA', 'NAMERICA', 'SAMERICA', 'WORLD']
    for region in regions:
        # get trends keys
        try:
            periods = MAP[region]['trends']['trends'].keys()
        except KeyError:
            print(region, '\t', 'Key Error')
            continue
        for p, per in enumerate(periods):
            r = MAP[region]['trends']['trends'][per]
            if p == 0:
                str_region = region[0:5]
                str_nmax = MAP[region]['nmax']
            else:
                str_region = ' '
                str_nmax = ' '
            if r['pval'] != None:
                if r['pval'] <= 0.1:
                    fstyle_start = bold_start
                    fstyle_end = bold_end
                if r['pval'] >= 0.1 and r['pval'] < 0.2:
                    fstyle_start = ''
                    fstyle_end = ''
                if r['pval'] > 0.2:
                    fstyle_start = italic_start
                    fstyle_end = italic_end
                print(str_region, '\t', per, '\t', fstyle_start, round(r['rel_slp'], 1), '±', round(
                    r['err_rel_slp'], 1), fstyle_end, '\t', round(r['pval'], 2))

            else:
                print(str_region, '\t', per, '\t',
                      'None', '±', 'None', '\t', 'None')
    print()


def col_region(region, alpha=1):
    if region == 'EUROPE':
        return (247 / 255, 163 / 255, 92 / 255, alpha)
    if region == 'ASIA':
        return (128 / 255, 133 / 255, 123 / 255, alpha)
    if region == 'AUSTRALIA':
        return (241 / 255, 92 / 255, 128 / 255, alpha)
    if region == 'NAFRICA':
        return (228 / 255, 211 / 255, 84 / 255, alpha)
    if region == 'SAFRICA':
        return (43 / 255, 144 / 255, 143 / 255, alpha)
    if region == 'NAMERICA':
        return (244 / 255, 91 / 255, 91 / 255, alpha)
    if region == 'SAMERICA':
        return (145 / 255, 232 / 255, 225 / 255, alpha)


def scat_trends(X_MAP, Y_MAP, params, obs_source, mod_source, 
                show_plot=True, save_plot=False):
    # plot that
    import seaborn as sns
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100,
                           facecolor='w', edgecolor='k')
    ax2 = plt.axes([.65, .2, .2, .2])
    xmin, xmax = -4, 2
    ymin, ymax = xmin, xmax

    xmin_outlier, xmax_outlier, ymin_outlier, ymax_outlier = [], [], [], []

    regions = ['EUROPE', 'ASIA', 'AUSTRALIA',
               'NAFRICA', 'SAFRICA', 'NAMERICA', 'SAMERICA']
    X, Y = [], []
    nout = 0
    for region in regions:
        # get trends of segments
        try:
            segs = list(X_MAP[region]['trends']['trends'].keys())
        except:
            # it means no trends were calculated for this region
            continue
        for i, seg in enumerate(segs):
            # obs
            x_rel_slp = X_MAP[region]['trends']['trends'][seg]['rel_slp']
            x_err_rel_slp = X_MAP[region]['trends']['trends'][seg]['err_rel_slp']

            if X_MAP[region]['trends']['trends'][seg]['pval'] != None:
                if X_MAP[region]['trends']['trends'][seg]['pval'] > 0.2:
                    x_rel_slp = None

            try:
                lst = list(Y_MAP[region]['trends']['trends'].keys())
            except KeyError:
                print('Key Error')
                continue
            if seg in lst:
                # mod
                y_rel_slp = Y_MAP[region]['trends']['trends'][seg]['rel_slp']
                y_err_rel_slp = Y_MAP[region]['trends']['trends'][seg]['err_rel_slp']

                if Y_MAP[region]['trends']['trends'][seg]['pval'] != None:
                    if Y_MAP[region]['trends']['trends'][seg]['pval'] > 0.2:
                        y_rel_slp = None

                # append in arrays in order to calculate correlation
                if ((x_rel_slp != None) and (y_rel_slp != None)):
                    X.append(x_rel_slp)
                    Y.append(y_rel_slp)

                    if x_rel_slp >= xmin and x_rel_slp <= xmax and y_rel_slp >= ymin and y_rel_slp <= ymax:
                        # plot in first axis
                        plt.sca(fig.axes[0])
                        plt.errorbar(x=x_rel_slp, y=y_rel_slp, xerr=x_err_rel_slp, yerr=y_err_rel_slp,
                                     fmt='o', color=col_region(region), alpha=1, markersize=6,
                                     elinewidth=1, ecolor=col_region(region, 0.8))

                        if seg != params['period']:
                            plt.sca(fig.axes[0])
                            plt.text(x=x_rel_slp, y=y_rel_slp + 0.025 * (ymax - ymin),  s='  ' + seg,
                                     horizontalalignment='left', verticalalignment='center', fontsize=8,
                                     color=col_region(region, alpha=0.8))

                        # plot points in second axis
                        plt.sca(fig.axes[1])
                        plt.errorbar(x=x_rel_slp, y=y_rel_slp, xerr=x_err_rel_slp, yerr=y_err_rel_slp,
                                     fmt='o', color=col_region(region), alpha=0.2, markersize=6,
                                     elinewidth=1, ecolor=col_region(region, 0.2))
                    else:
                        print('nout++')
                        nout += 1
                        # this is an inset axes over the main axes
                        xmin_outlier.append(x_rel_slp - x_err_rel_slp)
                        xmax_outlier.append(x_rel_slp + x_err_rel_slp)
                        ymin_outlier.append(y_rel_slp - y_err_rel_slp)
                        ymax_outlier.append(y_rel_slp + y_err_rel_slp)

                        if seg != params('period'):
                            plt.sca(fig.axes[1])
                            plt.text(x=x_rel_slp, y=y_rel_slp + 0.025 * (ymax - ymin),  s='  ' + seg,
                                     horizontalalignment='left', verticalalignment='center', fontsize=8,
                                     color=col_region(region, alpha=0.8))

                      # plot points in second axis
                        plt.sca(fig.axes[1])
                        plt.errorbar(x=x_rel_slp, y=y_rel_slp, xerr=x_err_rel_slp, yerr=y_err_rel_slp,
                                     fmt='o', color=col_region(region), alpha=1, markersize=6,
                                     elinewidth=1, ecolor=col_region(region, 0.8))

                    # plot in first axis
                    plt.sca(fig.axes[0])
                    plt.errorbar(x=x_rel_slp, y=y_rel_slp, xerr=x_err_rel_slp, yerr=y_err_rel_slp,
                                 fmt='o', color=col_region(region), alpha=0.2, markersize=6,
                                 elinewidth=1, ecolor=col_region(region, 0.8))

    # if no outlier, hide the second axe
    if nout == 0:
        fig.axes[1].set_visible(False)

    plt.sca(fig.axes[0])
    # plot diagonal
    plt.plot(np.linspace(xmin, xmax), np.linspace(ymin, ymax),
             color='darkgray', linewidth=1, linestyle='--')

    # plot zero lines
    plt.plot(np.linspace(xmin, xmax), 0 * np.linspace(ymin, ymax),
             color='darkgray', linewidth=1, linestyle=':')
    plt.plot(0 * np.linspace(xmin, xmax), np.linspace(ymin, ymax),
             color='darkgray', linewidth=1, linestyle=':')

    # plot correlation
    r = np.corrcoef(X, Y)
    # plt.text(x=xmin+0.1*(xmax-xmin), y=ymin+0.9*(ymax-ymin),  s = 'R: '+str(round(r[0][1],2)), horizontalalignment='left', verticalalignment='center', fontsize = 14, color = '#404040')

    ax.set_title(params['var'])
    ax.set_ylabel(mod_source + ' (%/yr)')
    ax.set_xlabel(obs_source + ' (%/yr)')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    dx = 0.5
    dy = dx
    # ax2.set_xlim(min(xmin_outlier)-dx,max(xmax_outlier)+dx)
    # ax2.set_ylim(min(ymin_outlier)-dy,max(ymax_outlier)+dy)
    
    
    #change face color of axis
    ax.set_facecolor('#F1F1F1')

    if save_plot:
        plt.savefig('figs/scatter_trends/' + params['var'] + '-' + \
            obs_source + '-' + mod_source + '-scatt.png', 
        dpi=600, bbox_inches='tight')

    if show_plot:
        plt.show()

def compute_trend_error(m, m_err, v0, v0_err):
    delta_sl = m_err / v0
    delta_ref = m * v0_err / v0**2
    return np.sqrt(delta_sl**2 + delta_ref**2) * 100



def plotTS(mmed, mbottom, mtop, ymed, trend, region, params, show_plot, save_plot):

    # import seaborn style
    sns.set()
    sns.set_context("paper")
    # sns.set_style("whitegrid")
    sns.color_palette("muted")
    # since each figure will be a subfigure, increase font_size
    fscale = 1.3
    sns.set(font_scale=fscale)

    # font style
    bold_start = r"$\bf{"
    bold_end = "}$"
    italic_start = r"$\it{"
    italic_end = "}$"

    # plot that
    fig, ax = plt.subplots(figsize=(9, 3), dpi=100,
                           facecolor='w', edgecolor='k')

    # plot monthly averages
    plt.fill_between(mmed.index, mbottom.values, mtop.values, alpha=.2)
    plt.plot(mmed, lw=2, alpha=.5)
    plt.plot(ymed, 'b.', lw=0, ms=12, alpha=1)
    plt.ylim(bottom=0)
    # plot every portions of the trend
    for nseg, seg in enumerate(trend.keys()):
        if len(trend[seg]['xdate']) > 0:
            if trend[seg]['pval'] <= 0.1:
                ls = '-'
                trend_style_start = bold_start
                trend_style_end = bold_end
                unit = '\ \%/yr '
                pm = '\ ±\ '
            if trend[seg]['pval'] > 0.1 and trend[seg]['pval'] <= 0.2:
                ls = '--'
                trend_style_start = ''
                trend_style_end = ''
                unit = ' %/yr '
                pm = ' ± '
            if trend[seg]['pval'] > 0.2:
                ls = ':'
                trend_style_start = italic_start
                trend_style_end = italic_end
                unit = '\ \%/yr '
                pm = '\ ±\ '
            # plot the trend
            plt.plot(trend[seg]['xdate'], trend[seg]['y'], lw=2,
                     linestyle=ls, color=sns.color_palette()[3], alpha=.5)

            # write the period
            string = seg + ': '
            xstr = 0.02
            ystr = 0.9 - 0.1 * nseg
            plt.text(xstr, ystr, string,
                     horizontalalignment='left', verticalalignment='center',
                     color=(.3, .3, .3), fontsize=8 * fscale,
                     transform=ax.transAxes
                     )

            # write the trend
            xstr = 0.15
            ystr = 0.9 - 0.1 * nseg
            string = trend_style_start + '{:.1f}'.format(trend[seg]['rel_slp']) + pm + '{:.1f}'.format(
                trend[seg]['err_rel_slp']) + unit + trend_style_end
            string += '; ' + italic_start + 'p' + italic_end + \
                '-val: ' + '{:.1e}'.format(trend[seg]['pval'])
            plt.text(xstr, ystr, string,
                     horizontalalignment='left', verticalalignment='center',
                     color=(.2, .2, .2), fontsize=9 * fscale,
                     transform=ax.transAxes,
                     )

            # plot vertival line
            if nseg > 0:
                plt.axvline(x=trend[seg]['xdate'][0], ls=':',
                            lw=2, color=(.5, .5, .5), alpha=.6)

    #ax.set_title(region+' '+period)
    if params['ylabel'] != None:
        ax.set_ylabel(params['ylabel'])
    else:
        ax.set_ylabel(var)
    if params['ymin'] != None:
        plt.ylim(bottom=params['ymin'])
    if params['ymax'] != None:
        plt.ylim(top=params['ymax'])
    # limit to period
    # plt.xlim(left=datetime.date(int(period.split('-')[0]),1,1))
    # plt.xlim(right=datetime.date(int(period.split('-')[1])+1,1,1))

    # write region at top-right
    ax.text(.97, .85, region, horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes, fontsize=14 * fscale)

    # remmove vertical grid
    sns.despine(left=True)

    #change face color of axis
    ax.set_facecolor('#F1F1F1')
    
    #set xaxis limit
    ax.set_xlim(params['period'].split('-')[0]+'-01-01',str(int(params['period'].split('-')[1])+1)+'-01-01')

    if save_plot:
        plt.savefig('figs/ts/OBS/' + params['source'] + '-' + params['var'] + '-' +
                    region + '.png', dpi=300, bbox_inches='tight')

    # show the plot
    if show_plot:
        plt.show()


def consistency(ref, exp, kind):
    if kind=='rel':
        diff = exp-ref
    elif kind=='abs':
        diff = (ref-exp)/ref
    norm = 1
    mu = 0
    stdv = 0.85
    gauss = 100*norm*np.exp(-0.5*(((diff-mu)/stdv)**2))
    
    consistency = gauss
    return consistency


def print_consistency(MOD, ALLTS_MOD, REG_MOD, kind='rel'):
    if kind=='rel':
        print('region', ' ', 'period', ' ', 'slp (%/yr)', ' ', 'allTS-slp (%/yr)', ' ', 'reg-slp (%/yr)', ' ', 'time_consist', ' ', 'space_consist',' ', 'consist')
        nsig = 1
        key_slope = 'rel_slp'
    elif kind=='abs':
        print('region', ' ', 'period', ' ', 'slp', ' ', 'allTS-slp', ' ', 'reg-slp', ' ', 'time_consist', ' ', 'space_consist',' ', 'consist')
        nsig = 4
        key_slope = 'abs_slp'
    print(' - - -', ' ', ' - - -', ' ', ' - - - - - - -', ' ', ' - - - - - - -', ' ', ' - - - - - - -', ' ', ' - - - - - -', ' ', ' - - - - - - ')
    regions = ['EUROPE', 'ASIA', 'AUSTRALIA', 'NAFRICA', 'SAFRICA', 'NAMERICA', 'SAMERICA']
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
            
            if p==0:
                str_region = region[0:5]
            else:
                str_region = ' '
            
            if r[key_slope]!=None and allts_r[key_slope]:
                slp = r[key_slope]
                allts_slp = allts_r[key_slope]
                reg_slp = reg_r[key_slope]    
                time_consist = consistency(slp, allts_slp, kind)
                space_consist = consistency(allts_slp, reg_slp, kind)
                all_consist = np.mean([time_consist, space_consist])
                
                str_slp = round(slp, nsig) #all colocated: time and space
                str_reg_slp = round(reg_slp, nsig)#no colocation at all: regoins, complete time series
                str_allts_slp = round(allts_slp, nsig) #spactial colocation, complete time series
                
            else:
                time_diff = np.nan
                space_diff = np.nan
                time_consist = np.nan
                space_consist = np.nan
                all_consist = np.nan
                accuracy = np.nan
                str_slp = None
                str_allts_slp = None
                str_reg_slp = None
                
            print(str_region, '\t', per , '\t', str_slp ,'\t', str_allts_slp , '\t', str_reg_slp , '\t',
                  #round(time_diff,1), '\t',
                  #round(space_diff,1), '\t',
                  '\033[1m', '{:.0f}'.format(round(time_consist,0)), '\t', 
                  '\033[1m', '{:.0f}'.format(round(space_consist,0)), '\033[0m', '\t',
                  '\033[1m', '{:.0f}'.format(round(all_consist,0)), '\033[0m' '\t',
                 )
            
    print()

    


def compute_lin_trend(x, y, params):
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
    
    delta_sl = slope_err/a
    
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

    # compute trend along time series
    return trend

def process_trend(data, params, obs=None, colocate_time=True,
                  colocate_space=True, OBS_DF=None, plot=True, write_json=False,
                  show_plot=False, save_plot=False):
    
    # by default, colocate model in space and time
    MAP, DF,  ALL_TS = {}, {}, {}
    regions = get_regions()

    # computation region by region
    for region in regions:
        f = pya.Filter(region)
        if params['kind'] == 'obs':
            try:
                sub = f(data)
                var = data.vars_to_retrieve[0]
                obs_var = var
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
            
            obs_var = obs.vars_to_retrieve[0]            
            
            #reduce data_all to list of filtered stations
            if region in OBS_DF:
                okstats = [col.split('_')[1] for col in list(OBS_DF[region].keys())]
                data_all = [data for data in data_all if data['station_name'] in okstats]
            else:
                data_all = []
            
            if colocate_space:
                # first, get list of stations name, lat and lon
                stations = {'name': [], 'lat': [], 'lon': []}

                for stat in data_all:
                    stations['name'].append(stat['station_name'])
                    stations['lat'].append(stat['station_coords']['latitude'])
                    stations['lon'].append(stat['station_coords']['longitude'])
                
                if len(data_all)>0:
                    data_all = data.to_time_series(
                        longitude=stations['lon'], latitude=stations['lat'],
                        add_meta=dict(station_name=stations['name'])
                    )
                else:
                    continue
                npoints = len(data_all)
            else:
                data_all = f(data)
                med_area = data_all.get_area_weighted_timeseries()
                npoints = np.shape(data_all.cube.data)[1]*np.shape(data_all.cube.data)[2]
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # initialize pandas DataFrame
        df = pd.DataFrame()

        stations = []
        if params['kind'] == 'mod' and colocate_space == False:
            med = med_area[var]
            top = med
            bottom = med
            ts = med.to_frame()
            ts.set_axis([obs_var + '_' + region], axis=1, inplace=True)
            df = pd.concat([df, ts], axis=1)
        else:
            for i, station in enumerate(data_all):
                stat_name = station.station_name
                print('region: ', region, 'station: ', stat_name, end="\r")
                ts_type = station.ts_type
                # set individual time series as dataframe

                # extract pandas series and convert it to datframe
                ts = data_all[i][var].to_frame()
                # remove duplicated index keeping the first occuence
                ts = ts.groupby(ts.index).first()
                #name the columns
                ts.set_axis([obs_var + '_' + stat_name], axis=1, inplace=True)
                #print(ts_type)
                if ts_type == 'hourly':
                    #make daily average
                    ts = ts.resample('D', how='mean')
                    # concatenates to main dataframe
                    #df = pd.concat([df, ts], axis=1)
                    if ts.count()[0] >= params['min_dobs']:
                        df = pd.concat([df, ts], axis=1)
                elif ts_type == 'daily':
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
        
        if params['kind'] == 'obs':
            #drop the time from datetime index by computing daily average
            df = df.resample('D', how='mean')

        # if model, colocates in time with obs
        if params['kind'] == 'mod' and colocate_time:
            if region in OBS_DF.keys():
                #if monthly values: set day to 1
                if params['min_dim'] == 0:
                    #print('min_dim==0')
                    '''
                    print('monthly data. set day to 1')
                    idx = df.index.values.astype('datetime64[M]')
                    df = df.set_index(idx)
                    df = df[OBS_DF[region].resample('D', how='mean') >= 0]
                    #df = df[OBS_DF[region]>= 0]
                    '''
                    print('colocate monthly dataframes')
                    df = df.resample('M', how='mean')
                    df_obs =  OBS_DF[region].resample('M', how='mean')
                    df = df[df_obs >= 0]
                else:
                    #if hourly or daily
                    df = df[OBS_DF[region].resample('D', how='mean') >= 0]
                #df = df[OBS_DF[region] >= 0]
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
        if len(xok) < params['min_ntrend']:
            print()
            print('Less than '+str(params['min_ntrend'])+' points in selected period')
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
        
        trend = compute_lin_trend(xok, yok, params)

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
        
        #npoints
        if params['kind'] == 'obs':
            nmax = len(df.columns)
        else:
            nmax = npoints

        # append to map dict
        MAP[region] = {
            'name': region,
            'min_lon': pya.Region(region).lon_range[0],
            'max_lon': pya.Region(region).lon_range[1],
            'min_lat': pya.Region(region).lat_range[0],
            'max_lat': pya.Region(region).lat_range[1],
            'trends': TS,
            'nmax': nmax,
            'stations': stations,
        }

        ALL_TS[region] = TS
        DF[region] = df

    # export MAP as json file
    if write_json:
        write_map(MAP, params)
        write_all_ts(ALL_TS, params)

    return TS, MAP, DF

def subplotTS(ax, il, ic, mmed, mbottom, mtop, ymed, trend, region, params):

    # import seaborn style
    sns.set()
    sns.set_context("paper")
    # sns.set_style("whitegrid")
    sns.color_palette("muted")
    # since each figure will be a subfigure, increase font_size
    fscale = 1.3
    sns.set(font_scale=fscale)

    # font style
    bold_start = r"$\bf{"
    bold_end = "}$"
    italic_start = r"$\it{"
    italic_end = "}$"

    # plot monthly averages
    ax.fill_between(mmed['x'], mbottom['y'], mtop['y'], alpha=.2)
    ax.plot(mmed['x'], mmed['y'], lw=2, alpha=.5)
    ax.plot(ymed['x'], ymed['y'], 'b.', lw=0, ms=12, alpha=1)
    ax.set_ylim(bottom=0)
    
    # plot every portions of the trend
    for nseg, seg in enumerate(trend.keys()):
        if len(trend[seg]['jsdate']) > 0:
            if trend[seg]['pval'] <= 0.1:
                ls = '-'
                trend_style_start = bold_start
                trend_style_end = bold_end
                unit = '\ \%/yr '
                pm = '\ ±\ '
            if trend[seg]['pval'] > 0.1 and trend[seg]['pval'] <= 0.2:
                ls = '--'
                trend_style_start = ''
                trend_style_end = ''
                unit = ' %/yr '
                pm = ' ± '
            if trend[seg]['pval'] > 0.2:
                ls = ':'
                trend_style_start = italic_start
                trend_style_end = italic_end
                unit = '\ \%/yr '
                pm = '\ ±\ '
            # plot the trend
            ax.plot(js2date(trend[seg]['jsdate']), trend[seg]['data'], lw=2,
                     linestyle=ls, color=sns.color_palette()[3], alpha=.5)
            

            # write the period
            string = seg + ': '
            xstr = 0.02
            ystr = 0.9 - 0.1 * nseg
            ax.text(xstr, ystr, string,
                     horizontalalignment='left', verticalalignment='center',
                     color=(.3, .3, .3), fontsize=8 * fscale,
                     transform=ax.transAxes
                     )

            # write the trend
            xstr = 0.15
            ystr = 0.9 - 0.1 * nseg
            string = trend_style_start + '{:.1f}'.format(trend[seg]['rel_slp']) + pm + '{:.1f}'.format(
                trend[seg]['err_rel_slp']) + unit + trend_style_end
            string += '; ' + italic_start + 'p' + italic_end + \
                '-val: ' + '{:.1e}'.format(trend[seg]['pval'])
            ax.text(xstr, ystr, string,
                     horizontalalignment='left', verticalalignment='center',
                     color=(.2, .2, .2), fontsize=9 * fscale,
                     transform=ax.transAxes,
                     )

            # plot vertival line
            if nseg > 0:
                ax.axvline(x=js2date(trend[seg]['jsdate'])[0], ls=':',
                            lw=2, color=(.5, .5, .5), alpha=.6)

    #for first column, add parameter as ylabel
    #if (ic==0):
    #    ax.set_ylabel(params['ylabel'],ha='left')
    if params['ymin'] != None:
        ax.set_ylim(bottom=params['ymin'])
    if params['ymax'] != None:
        ax.set_ylim(top=params['ymax'])

    # write region at top-right
    ax.text(.95, .9, region, ha='right', va='top', transform=ax.transAxes, fontsize=12 * fscale)

    # remmove vertical grid
    sns.despine(left=True)

    #change face color of axis
    ax.set_facecolor('#F1F1F1')
    
    #set xaxis limit
    y1 = int(params['period'].split('-')[0])
    y2 = int(params['period'].split('-')[1])
    ax.set_xlim([datetime.date(y1, 1, 1), datetime.date(y2+1, 1, 1)])