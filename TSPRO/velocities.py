import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from scipy.optimize import curve_fit
import shapefile as shp
from numpy.linalg import inv
from matplotlib.patches import Ellipse

import constants

# change hight/width ratio (because of geografical position of SVK - fi, la)
WIDTH_ASPECT = np.deg2rad(49)
# WIDTH_ASPECT = 1

def station_preparation(station, direc='time_series_export', interp=False, plot_interpol=False, xyz=True, gps_week_idx=True, use_SDs=True):
    '''
    nacitanie dat, zoradenie a pripadna interpolacia (linearna) zo suboru time_series_STAT.csv (STAT-station)

    INPUT
    station - nazov stanice (string)
    direc - priecinok v cwd s datami
    plot_interpol - ci sa maju vykreslit vyinterpolovane data 
    xyz = True - nacitavaju sa suradnice x,y,z
    xyz = False - nacitavaju sa suradnice n,e,u
    gps_week_idx: bool; True - index is GPS week
                        Fasle - index is date
    interp: bool; linear interpolation of missing values (valid only for gps_week_idx=True)

    OUTPUT
    dataframe s datetime indexom + 3 suradnice
    '''
    # nazov stanice malym
    file_path = '{}/{}.csv'.format(direc,station)
    if not os.path.exists(file_path):
        return None

    if xyz == True:
        coors = ['x', 'y', 'z']
    else:
        coors = ['n', 'e', 'u']

    if gps_week_idx:
        date_index = 'gps_week'
    else:
        date_index = 'date'

    # loading data
    data = pd.read_csv(file_path, names = ['value_id','date(YYYYMMDD)','year(YYYY.YYYYY)','gps_week','n(m)',
                                                                          'n_std_dev(m)','e(m)','e_std_dev(m)','u(m)','u_std_dev(m)','x(m)',
                                                                          'x_std_dev(m)','y(m)','y_std_dev(m)','z(m)','z_std_dev(m)','coord_type'])
    # zoradenie podla GPS week
    if date_index == 'date':
        data['date'] = pd.to_datetime(data['date(YYYYMMDD)'], format="%Y%m%d")

    data = data.sort_values(date_index)
    data = data.reset_index(drop=True)
    # skontrolovanie ci mam spravne data (ak je tam len sinex)
    # print(data['coord_type'].unique())
    if not len(data['coord_type'].unique()) == 1:
        print('pozor, data obsahuju rozne zdroje')
        return None
    if use_SDs:
        data = data[[date_index,'{}(m)'.format(coors[0]),'{}(m)'.format(coors[1]),'{}(m)'.format(coors[2]),
                    '{}_std_dev(m)'.format(coors[0]), '{}_std_dev(m)'.format(coors[1]), '{}_std_dev(m)'.format(coors[2]) ]]
    else:
        data = data[[date_index,'{}(m)'.format(coors[0]),'{}(m)'.format(coors[1]),'{}(m)'.format(coors[2])]]

    if interp:
        # prva a posledna hodnota gps tyzdna
        w1 = data['gps_week'].iloc[0]
        w2 = data['gps_week'].iloc[-1]
        # vytvorenie dataframeu s maximalnou dlzkou
        df = pd.DataFrame(np.arange(w1,w2+1),columns=['gps_week'])
        # spojenie dvoch ramcov : vytvorenie SQL-like outer joinu (bud xyz, alebo neu)
        df = df.merge(data, left_on='gps_week',right_on='gps_week',how='outer')

        ### GPSweek ako index
        df = df.set_index('gps_week')
        #linearna interpolacia chybajucich hodnot
        df = df.interpolate()
        # ##df = df.interpolate(method='quadratic')
        # ##df = df.interpolate(method='cubic')
        # ##df = df.interpolate(method='spline',order=3,s=0.)

        # vykreslenie interpolacie jednej zo suradnic
        if plot_interpol:
            plt.plot(df.index,df[coors[0]].values,c='b',marker='o',linestyle='--',label='interpolated data')
            plt.plot(data['gps_week'].values,data['{}(m)'.format(coors[0])].values,c='r',marker='o',linestyle='--',label='orig. data')
            plt.legend(loc='best')
            plt.title('interpolacia hodnot')
            plt.show()
    else:
        df = data.set_index(date_index)

    # nazvy stlpcov
    if use_SDs:
        df.columns = coors + [ 'SD_{}'.format(i) for i in coors]
    else:
        df.columns = coors
    return df


def get_blocks(seq):
    '''
    gets blocks of text file as generator. (FODITS output)
    '''
    correct_block = False

    data = []
    for line in seq:
        if line.startswith(' OUTPUT: A POSTERIORI INFORMATION - ESTIMATES'):
            correct_block = True
        elif line.startswith(' SUMMARY OF RESULTS'):
            break

        if correct_block:
            if line.startswith('  Nr Station'):
                if data:
                    yield data
                    data = []
            data.append(line)


def get_blockstats(block):
    '''
    Get station name and dates of discontinuities for one station (FODITS output)
    block: block of text

    returns (station, list of dates)
    '''
    discs = []
    station = None
    for line in block:
        splitted_line = line.split()
        if len(splitted_line) == 16:
            if station is None:
                station = splitted_line[1]
            event = splitted_line[3]
            signif = splitted_line[-2]
            if event == 'DISC' and signif == 'Y':
                disc_date = splitted_line[5]
                discs.append(disc_date)
    return (station, discs)


def get_Bern_discontinuities(file):
    '''
    Get list of discontinuities for every station from FODITS output file.
    file: output file from FODITS
    disconties: dict. with list of dates (date of discontinuitie) for every station
    '''
    disconties = {}
    with open(file) as obj:
        gen = get_blocks(obj)
        for block in gen:
            station, discs = get_blockstats(block)
            if station is not None:
                disconties[station] = discs
    return disconties


def create_custom_discofile(file, discons):
    '''
    Create file with dates of discontinuities from FODITS for custom manipulation.

    file: str; file to write
    discons: dict. with list of dates (date of discontinuitie) for every station
                - got from FODITS from Bernese
    '''
    with open(file, 'w') as obj:
        for station in sorted(discons):
            disco_dates = ','.join(discons[station])
            obj.write('{},{}\n'.format(station, disco_dates))


def load_discofile(file):
    '''
    Load dates of discontinuities from the custom (modified) file.
    '''
    new_discos = {}
    with open(file) as obj:
        for line in obj:
            if line.startswith('#'):
                continue
            station, *dates = line.strip().split(',')
            if len(line) > 6:
                new_discos[station] = dates
            else:
                new_discos[station] = []
    return new_discos


def load_exclfile(file):
    '''
    Load date intervals for exclusion from custom file.
    '''
    excl_intervals = {}
    with open(file) as obj:
        for line in obj:
            if line.startswith('#'):
                continue
            station, *dates = line.strip().split(',')
            excl_intervals.setdefault(station,[]).append(dates)
    return excl_intervals


def linefit_on_datetime(dt_index, y):
    '''
    Fit y-values by line and get velocity for one year

    INPUTS
    datetime_index: pd.DatetimeIndex
    y: float; values for fitting

    OUTPUTS
    trendline: array; estimated values for every x-coorinate (dt_index)
    change_yearly: velocity for one year (m/year if y input is in meters)
    slope: float; slope of fitted line
    '''
    # constant to convert ns to years (estimated slope will be automatically velocity m/year)
    nano2year = 1e+9*60*60*24*365
    # convert date to float
    X = dt_index.astype(int)/nano2year
    # p - Polynomial coefficients, highest power first
    # V - covariance matrix
    p, V = np.polyfit(x=X, y=y, deg=1, cov=True)
    f = np.poly1d(p)
    trendline = f(X)
    # get velocity for one year
    year_range = pd.date_range(datetime.date(2017,1,1), periods=2, freq='365D')
    # same transformation as in first step
    year_transed = year_range.astype(int)/nano2year
    ys_for_year = f(year_transed)
    # using nano2year constant this should be the same as slope
    change_yearly = ys_for_year[1] - ys_for_year[0]
    slope = p[0]
    sigma_slope = np.sqrt(V[0][0])

    return trendline, change_yearly, slope, sigma_slope


def plot_triple_coors(df, coors, title, draw=False, plot_trends=True, new_figure=True, discos=None):
    '''
    Plot 3 subplots with same x-axis

    df: df with n,e,u columns
    discos: list of strings; dates of discontinuities. If not None, they are 
    used to separate plot lines (vizualization purpose).
    '''
    df_new = df.copy()
    if (discos is not None) and (len(discos) > 0):
        df_empty = pd.DataFrame(index=pd.to_datetime(discos))
        df_new = pd.concat([df_new, df_empty])
        df_new.sort_index(inplace=True)

    if coors[0] == 'n':
        l0, l1, l2 = [0.01, 0.01, 0.02]
    else:
        l0, l1, l2 = [0.015, 0.015, 0.015]

    if new_figure:
        plt.figure()

    ax1 = plt.subplot(311)
    mean0 = df_new[coors[0]].mean()
    plt.plot(df_new[coors[0]]-mean0)
    if plot_trends:
        trend_col0 = 'trend_{}'.format(coors[0])
        plt.plot(df_new[trend_col0]-mean0, linewidth=2.0)
    plt.setp(ax1.get_xticklabels(), fontsize=6, visible=False)
    plt.grid()
    plt.title(title)
    plt.ylabel('{} [m]'.format(coors[0]))
    # plt.ylim([df_new[coors[0]].mean() - l0, df_new[coors[0]].mean() + l0])
    plt.ylim([0 - l0, 0 + l0])

    # share x only
    ax2 = plt.subplot(312, sharex=ax1)
    mean1 = df_new[coors[1]].mean()
    plt.plot(df_new[coors[1]] - mean1)
    if plot_trends:
        trend_col1 = 'trend_{}'.format(coors[1])
        plt.plot(df_new[trend_col1] - mean1, linewidth=2.0)
    # make these tick labels invisible
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.grid()
    plt.ylabel('{} [m]'.format(coors[1]))
    plt.ylim([0 - l1, 0 + l1])

    # share x and y
    ax3 = plt.subplot(313, sharex=ax1)
    mean2 = df_new[coors[2]].mean()
    plt.plot(df_new[coors[2]] - mean2)
    if plot_trends:
        trend_col2 = 'trend_{}'.format(coors[2])
        plt.plot(df_new[trend_col2] - mean2, linewidth=2.0)
    plt.grid()
    plt.ylabel('{} [m]'.format(coors[2]))
    plt.xlabel('year')
    plt.ylim([0 - l2, 0 + l2])
    if draw:
        plt.draw()
    else:
        plt.show()


def get_vel_res_B(ref_vel, sol_vel):
    '''
    Returns df with velocity solutions in ITRF and velocity residuals (solution-ref)

    ref_vel: *.VEL file with reference velocities from NUVEL model for every station and subnames
                - it is result from FODITS
    sol_vel: *.VEL file with final velocities from ADDNEQ (FODITS files were inputs in ADDNEQ)
    '''
    header_names = ['ID_stat', 'station', 'DOMES', 'Vx(m/y)', 'Vy(m/y)', 'Vz(m/y)', 'flag', 'plate']
    df_ref = pd.read_csv(ref_vel, skiprows=6, delim_whitespace=True, names=header_names)
    # some short intervals dont have solution
    df_solution = pd.read_csv(sol_vel, skiprows=6, delim_whitespace=True, names=header_names)
    df_red = df_solution[['Vx(m/y)', 'Vy(m/y)', 'Vz(m/y)']] - df_ref[['Vx(m/y)', 'Vy(m/y)', 'Vz(m/y)']]
    df_solution[['Vx(m/y)res', 'Vy(m/y)res', 'Vz(m/y)res']] = df_red
    return df_solution


def fit_subTSs_separ(df, discos, coors, outlier_trash=3, period_year=True):
    '''
    Split 3-coords time series on discontinuities and fit a line on each subTS.
    Returns estimated slopes for each part and new positions of points (trendline) 
    in original df.
    
    INPUTS
    df: station coordinates loaded and prepared
    discos: list of strings; dates of discontinuities for station

    OUTPUT
    df: original df with added slope and trend column for all coordinates
    '''
    # splitting on discontinuities
    # add smallest and biggest border
    df_parts = []
    date_borders = ['1980-01-01'] + discos.copy() + ['2050-01-01']
    for i in range(len(date_borders) - 1):
        left_border = date_borders[i]
        right_border = date_borders[i+1]
        mask = (df.index > left_border) & (df.index < right_border)
        # group numbering
        df.ix[mask, 'part'] = i
        df_part = df[mask]

        # in case we have more actual dates of jumps from Bernese then input station coordinates
        if len(df_part) == 0:
            continue

        masks_no_outliers = []
        for coor in coors:
            theta, A, Cova, RMSE = MNS_w_jumps(df_part[coor].values, df_part.index, [], period_year=period_year)
            trendline = np.dot(A, theta).ravel()
            slope = theta[0,0]
            # trendline, change_yearly, slope, sigma_slope = linefit_on_datetime(df_part.index, df_part[coor])
            # print('slope', slope)
            # print('sigma_slope', sigma_slope)
            trend_col = 'trend_{}'.format(coor)
            df.ix[mask, trend_col] = trendline
            slope_col = 'slope_{}'.format(coor)
            df.ix[mask, slope_col] = slope

            # detect outliers
            # outlier in one TS means excludion of point from all 3 TS
            if outlier_trash is not None:
                mask_in = (df_part[coor] > (df.ix[mask, trend_col] - outlier_trash*RMSE)) & (df_part[coor] < (df.ix[mask, trend_col] + outlier_trash*RMSE))
                masks_no_outliers.append(mask_in)

        if outlier_trash is not None:
            mask_common = masks_no_outliers[0] & masks_no_outliers[1] & masks_no_outliers[2]
            df_part_cleaned = df_part.copy()[mask_common]

            for coor in coors:
                theta, A, Cova, RMSE = MNS_w_jumps(df_part_cleaned[coor].values, df_part_cleaned.index, [], period_year=period_year)
                col_name = 'trend_{}'.format(coor)
                df_part_cleaned.loc[:,col_name] = np.dot(A, theta)
                slope_col = 'slope_{}'.format(coor)
                df_part_cleaned[slope_col] = theta[0,0]
                # rewrite original values
            df_parts.append(df_part_cleaned)

    if outlier_trash is not None:
        df_final = pd.concat(df_parts)
    else:
        df_final = df

    return df_final


def fit_fullTS_w_jumps(df, discos, coors, outlier_trash=3, plot_outliers=False, period_year=True, max_iter=2, use_weights=True):
    '''
    Fit a line to full TS with estimation of jumps in one step based on MNS (OLS) for all 3 coordinates.
    https://www.irsm.cas.cz/materialy/acta_content/2016_doi/Rapinski_AGG_2016_0013.pdf

    urcujuca rovnoca: y = ax + b + C1*j1 + C2*j2 + ...
    theta = [a b j1 j2 ...]

    INPUTS
    df: station coordinates loaded and prepared
    outlier_trash: int or None; if None - no outlier removal
                                if int - number of times of RMSE as treshold for outlier removal
    coors: list of str; coordinates to process
    discos: list of strings; dates of discontinuities for station
    plot_outliers: bool; plot TS w/wo outliers for comparsion
    period_year: bool; includ yearly period in OLS model
    max_iter: int; max number of iterations (new model fitting) for outlier detection
    use_weights: bool;  if True - individual sigmas of observations will be used as weight matrix 
                                    in estimation model (WLS instead OLS).
                        if False - no weights are used.


    OUTPUTS
    df_final: same df as input, with new column added - new estimated values. This df will be 
                shorter if outliers are detected.
    yearly_vels: dict.; velocities m/year for all 3 coordinates
    '''
    yearly_vels = {}
    yearly_vels_SDs = {}
    sigmas_0_coors = {}
    for coor in coors:
        if use_weights:
            priorSD_name = 'SD_{}'.format(coor)
            prior_sigmas = df[priorSD_name].values
        else:
            prior_sigmas = None
        theta, A, Cova, RMSE = MNS_w_jumps(df[coor].values, df.index, discos, period_year=period_year, weights=prior_sigmas)
        col_name = 'trend_{}'.format(coor)
        df[col_name] = np.dot(A, theta)
        yearly_vels[coor] = theta[0,0]
        yearly_vels_SDs[coor] = np.sqrt((Cova[0,0]))
        sigmas_0_coors[coor] = RMSE
        df_final = df

    # detect outliers
    # outlier in one TS means excludion of point from all 3 TS
    if outlier_trash is not None:
        df_cleaned = df.copy()
        for i in range(max_iter):
            df_cleaned, n_outs = detect_outliers(df_cleaned, outlier_trash, coors, sigmas_0_coors)
            if n_outs == 0:
                break

            for coor in coors:
                if use_weights:
                    priorSD_name = 'SD_{}'.format(coor)
                    prior_sigmas = df_cleaned[priorSD_name].values
                else:
                    prior_sigmas = None
                theta, A, Cova, RMSE = MNS_w_jumps(df_cleaned[coor].values, df_cleaned.index, discos, period_year=period_year, weights=prior_sigmas)
                sigmas_0_coors[coor] = RMSE
                col_name = 'trend_{}'.format(coor)
                df_cleaned.loc[:,col_name] = np.dot(A, theta)
                # rewrite original values
                yearly_vels[coor] = theta[0,0]
                yearly_vels_SDs[coor] = np.sqrt((Cova[0,0]))
                df_final = df_cleaned

        # plot fitting lines for both cases 
        if plot_outliers:
            plot_triple_coors(df, coors, '', draw=True)
            plot_triple_coors(df_cleaned, coors, 'w/wo outliers', draw=False, new_figure=False)

    return df_final, yearly_vels, yearly_vels_SDs


def detect_outliers(df_in, outlier_trash, coors, sigmas_0_coors):
    masks_no_outliers = []
    for coor in coors:
        col_name = 'trend_{}'.format(coor)
        sigma = sigmas_0_coors[coor]
        mask_in = (df_in[coor] > (df_in[col_name] - outlier_trash*sigma)) & (df_in[coor] < (df_in[col_name] + outlier_trash*sigma))
        masks_no_outliers.append(mask_in)

    mask_common = masks_no_outliers[0] & masks_no_outliers[1] & masks_no_outliers[2]
    df_cleaned = df_in.copy()[mask_common]
    n_outs = len(df_in) - len(df_cleaned)
    return df_cleaned, n_outs


def mdot(*args):
    ''' np.dot for more matrices from left to right'''
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret,a)
    return ret


def MNS_w_jumps(x, time_index, discos, period_year=True, Ampl0=0.003, phase0=0, weights=None):
    '''
    Fit a line on array with jumps.
    source: https://www.irsm.cas.cz/materialy/acta_content/2016_doi/Rapinski_AGG_2016_0013.pdf

    x: np array; all observations
    time_index: datetiem index from df
    discos: list od strings; dates od discontinuities
    weights: array of floats or None; prior SDs for observations. 
                if None - all observations will have the same weights
                if array - weight matrix P will be used. 

    urcujuca rovnica: y = ax + b + Apmli*sin(2pi*x/1) + C1*j1 + C2*j2 + ... 
    theta = [a b Apmli j1 j2 ... ]
    '''

    if weights is None:
        P =  np.diag(np.ones(len(x)))
    else:
        real_weights = np.ones(len(x)) / weights**2
        real_weights = real_weights / real_weights[0]
        P = np.diag(real_weights)

    x = x.reshape(-1,1)
    n = len(x) # n. observations
    k = 2 + len(discos) # num. of estimated params
    nano2year = 1e+9*60*60*24*365
    # convert date to float
    times = (time_index.astype(int)/nano2year).values.reshape(-1,1)
    ones = np.ones(len(x)).reshape(-1,1)
    if period_year:
        ampl_col = np.sin(times*2*np.pi/1 + phase0)
        phase_col = Ampl0 * np.cos(times*2*np.pi/1 + phase0)
        A = np.concatenate((times, ones, ampl_col, phase_col), axis=1)
    else:
        # matica planu if no jumps presented
        A = np.concatenate((times, ones), axis=1)
    if len(discos) > 0:
        # create matrix C ("step matrix")
        n_jumps = len(discos)
        C = np.zeros((n_jumps, len(x)))
        for i, jump_date in enumerate(discos):
            mask = time_index > jump_date
            C[i, mask] = 1
        A = np.concatenate((A, C.T), axis=1)

    # theta = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), x)
    first_p = inv(mdot(A.T, P, A))
    theta = mdot(first_p, A.T, P, x)

    # opravy
    v = np.dot(A, theta) - x
    # sigma0_sq = np.dot(v.T, v) / (n - k)
    sigma0_sq = mdot(v.T, P, v) / (n - k)
    Cova = sigma0_sq * inv(mdot(A.T, P, A))
    RMSE_sq = np.dot(v.T, v) / (n - k)
    RMSE = np.sqrt(RMSE_sq)[0,0]
    return theta, A, Cova, RMSE


def estimate_intercept(X, Y, slope):
    '''
    Estimate intercept for line fitting ( b in eq. y = ax + b).
    Rreturns array with estimated values of fitted line.
    '''

    def f_fixed_slope(x, B):
        return slope*x + B

    intercept = curve_fit(f_fixed_slope, X, Y)[0][0]
    f = np.poly1d([slope, intercept])
    trendline = f(X)
    return trendline, intercept


def prepare_triple_graph(df, coors, title):
    '''
    Prepare triple figure for coordinates with labels, title, grid...
    Returns dictionary for mapping coordinates name to the figure axis.
    '''    
    if coors[0] == 'n':
        l0, l1, l2 = [0.01, 0.01, 0.02]
    else:
        l0, l1, l2 = [0.015, 0.015, 0.015]

    plt.figure()
    ax1 = plt.subplot(311)
    plt.grid()
    plt.title(title)
    plt.ylabel('{} [m]'.format(coors[0]))
    plt.ylim([df[coors[0]].mean() - l0, df[coors[0]].mean() + l0])
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(312, sharex=ax1)
    plt.grid()
    plt.ylabel('{} [m]'.format(coors[1]))
    plt.ylim([df[coors[1]].mean() - l1, df[coors[1]].mean() + l1])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3 = plt.subplot(313, sharex=ax1)
    plt.grid()
    plt.ylabel('{} [m]'.format(coors[2]))
    plt.ylim([df[coors[2]].mean() - l2, df[coors[2]].mean() + l2])

    coor2ax = dict(zip(coors, [ax1, ax2, ax3]))
    return coor2ax


def get_averaged_velo(df, coors, station, plots=True):
    '''
    Get averaged velocity (m/year) for all 3 coordinates for one station.
    Plot weighted averaged slope of subTS.
    The coordinate will have the same slope (averaged subslopes).

    INPUTS
    df: dataframe with 3 coordinates and slopes for every subTS
    coors: list of coordinates names
    plots: bool; to plot/ not to plot AVG slope for original data
    '''
    # constant to convert ns to years (estimated slope will be automatically speed)
    nano2year = 1e+9*60*60*24*365
    yearly_vels = {}
    if plots:
        coor2ax = prepare_triple_graph(df, coors, '{}, AVGed slopes fitted separately'.format(station))

    for coor in coors:
        slope_col = 'slope_{}'.format(coor)
        # weighted average
        AVG_slope = df[slope_col].mean()
        # # in case we are using nano2year constant, AVG_slope is automaticaly yearly_vel
        # yearly_vel = get_velocity_yearly(df, AVG_slope)
        yearly_vels[coor] = AVG_slope

        if plots:
            plt.sca(coor2ax[coor])
            for name, df_group in df.groupby('part'):
                # estimate intercept for every subplot for better visualization.
                group_line, intercept = estimate_intercept(df_group.index.astype(int)/nano2year, df_group[coor].values, AVG_slope)
                plt.plot(df_group.index, group_line)
            plt.plot(df.index, df[coor], c='black')

    if plots:
        plt.show()
    
    return yearly_vels


def get_velocity_yearly(df, slope):
    '''
    in case we are using nano2year constant, slope is automaticaly yearly_vel
    '''
    nano2year = 1e+9*60*60*24*365
    f = np.poly1d([slope, 0])
    # get velocity for one year
    year_range = pd.date_range(datetime.date(2017,1,1), periods=2, freq='365D')
    # same transformation as in first step
    year_transed = year_range.astype(int)/nano2year
    ys_for_year = f(year_transed)
    change_yearly = ys_for_year[1] - ys_for_year[0]
    return change_yearly


def plot_SVK(shp_file, draw=False):
    plt.figure()
    sf = shp.Reader(shp_file)
    # SVK is 44
    state_id = 44
    shapeRecs = sf.shapeRecords()
    points = shapeRecs[state_id].shape.points[0:2]

    plt.gca().set_aspect(1/WIDTH_ASPECT)
    x = [i[0] for i in shapeRecs[state_id].shape.points[:]]
    y = [i[1] for i in shapeRecs[state_id].shape.points[:]]
    plt.plot(x,y, c='black')
    if draw:
        plt.draw()
    else:
        plt.show()


def exclude_short_parts(df, discos, weeks_tresh_part=65, weeks_tresh_all=156):
    '''
    Add part column based on discontinuity dates and exclude 
    intervales shorter then treshold (weeks_tresh_part) and weeks_tresh_all

    df: laoded station coordinates
    discos: list of strings; dates of discontinuities
    weeks_tresh_part: int; intervals shorter then this number of weeks will be excluded
    weeks_tresh_all: int; minimum of weeks in final df (after subTS exclusion)
    '''
    df_new = df.copy()
    excluded_groups = []
    # add smallest and biggest border
    date_borders = ['1980-01-01'] + discos.copy() + ['2050-01-01']
    for i in range(len(date_borders) - 1):
        left_border = date_borders[i]
        right_border = date_borders[i+1]
        mask = (df_new.index > left_border) & (df_new.index < right_border)
        # group numbering
        df_new.ix[mask, 'part'] = i
        if sum(mask) < weeks_tresh_part:
            excluded_groups.append(i)

    # shorter subTS then this number of weeks will be excluded
    df_excl = df_new.groupby('part').filter(lambda x: len(x) >= weeks_tresh_part)
    if len(df_excl) == 0:
        return None, None

    # modify dates of discontinuities in case of exclusion
    if len(excluded_groups) == 0:
        discos_mod = discos
    else:
        discos_mod = [ element for j, element in enumerate(discos) if j not in excluded_groups]
        # if last subTS is short last date is excluded
        if excluded_groups[-1] == len(discos):
            discos_mod = discos_mod[:-1]

    if len(df_excl) < weeks_tresh_all:
        return None, None

    return df_excl, discos_mod


def exclude_custom_intervals(df, excl_inters, station):
    '''
    Exclude date intervals from df.
    '''
    df_new = df.copy()
    if not station in excl_inters:
        return df_new

    ex_intervals = excl_inters[station]
    final_mask = np.ones(len(df))
    for interval in ex_intervals:
        start_date = interval[0]
        end_date = interval[1]
        mask = (df_new.index >= start_date) & (df_new.index <= end_date)
        final_mask[mask] = 0

    final_mask = final_mask.astype(bool)
    df_new = df_new[final_mask]
    return df_new


def plot_error_ellipse(ax, center, sigma_x, sigma_y):
    '''
    Draw error ellipse with 95% confident interval based on sigma x and y (no covariance)
    with two degrees of freedom.
    http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

    ax: plt axis where to draw
    center: tuple with coordinates x,y;
    sigma_x, sigma_y: standard deviations
    '''
    # WIDTH_ASPECT is used to correct changed hight/width ratio
    conf_int95 = 5.991 # two DOF
    ell = Ellipse(xy=center, width=sigma_x*2*np.sqrt(conf_int95), 
                height=sigma_y*2*np.sqrt(conf_int95)*WIDTH_ASPECT, color='r')
    ell.set_facecolor('none')
    ax.add_artist(ell)


def get_velos_df(COORS_FILE, stations_velos, stations_velos_SDs):
    ''''Save station coordinates, velocities and SDs to csv file
    '''
    def long2short(value):
        deg = int(value[:2])
        mins = int(value[3:5])
        seks = float(value[6:-1])
        return deg + mins/60 + seks/3600


    def velos2df(row, velos, coor='n'):
        station = row['station']
        if station in velos:
            result = velos[station][coor]
        else:
            result = np.nan
        return result

    df = pd.read_csv(COORS_FILE, delim_whitespace=True)
    df['fi'] = df['fi'].apply(long2short)
    df['la'] = df['la'].apply(long2short)
    df['v_n'] = df.apply(velos2df, axis=1, args=(stations_velos,), coor='n')
    df['v_e'] = df.apply(velos2df, axis=1, args=(stations_velos,), coor='e')
    df['v_u'] = df.apply(velos2df, axis=1, args=(stations_velos,), coor='u')
    df['Sv_n'] = df.apply(velos2df, axis=1, args=(stations_velos_SDs,), coor='n')
    df['Sv_e'] = df.apply(velos2df, axis=1, args=(stations_velos_SDs,), coor='e')
    df['Sv_u'] = df.apply(velos2df, axis=1, args=(stations_velos_SDs,), coor='u')
    return df



def get_final_velocities(DST_FILE, EXCL_FILE, TS_DIR, plot_each_fit=False, plot_each_outliers=False,
                        period_year=True, stat=None, outlier_trash=3, fit_fullTS=True, 
                        weeks_tresh_part=65, weeks_tresh_all=156, plot_each_res=False, use_weights=True):
    '''
    Get yearly velocities for all stations defined in DST_FILE. (m/year) and sigmas

    INPUTS
    DST_FILE: str; file with dates of discontinuities
    EXCL_FILE: str; file with custom dates for exclusion
    TS_DIR: str; directory which contain coordinats of each station (*.csv files)
    plot_each_fit: bool; plot fitted model for each station separately
    outlier_trash: int or None; if None - no outlier removal
                                if int - number of times of RMSE as treshold for outlier removal
    plot_each_outliers: bool; plot outlier removal process for each station and coordinate
    fit_fullTS: bool;   True - fit full TS with linear trend (prefered)
                        False - fit each part separately and AVG velocities
    period_year: bool; True - include one year period in estimation
    stat: str or None;  if str - station name for processing
                        if None - all stations are processed
    use_weights: bool;  if True - individual sigmas of observations will be used as weight matrix 
                                    in estimation model (WLS instead OLS).
                        if False - no weights are used.

    OUTPUTS
    stations_velos, stations_velos_SDs: dic; velocities and sigmas for all stations for all coordinates
    stations_dfres: dict of dfs; 
    '''
    coors = ['n', 'e', 'u']

    # # dates of discontinuities for all stations from bernese
    # bern_discons = get_Bern_discontinuities(constants.FODITS_FILE)
    # # write custom "STA" file with dates of discontinuities
    # create_custom_discofile('new.dst', bern_discons)

    # load custom "STA" file
    discons = load_discofile(DST_FILE)
    excl_inters = load_exclfile(EXCL_FILE)
    # print(excl_inters)

    # # residual velocities from Bernese solution
    # Bern_velos = get_vel_res_B(constants.VEL_FILE_REF, constants.VEL_FILE_SOL)

    # velocities for all stations
    stations_velos = {}
    stations_velos_SDs = {}
    stations_dfres = {}
    for station, discos in discons.items():
        if stat is not None:
            if station != stat:
                continue

        df = station_preparation(station, direc=TS_DIR, interp=False, xyz=False, gps_week_idx=False)
        if df is None:
            print('stanica {} bola preskocena'.format(station))
            continue
        else:
            print(station)
            # print('discos orig.: ', discos)

        # df = df[:'2012-12-24']
        # if len(df) == 0:
        #     continue

        # exclude custom intervals from TS
        df = exclude_custom_intervals(df, excl_inters, station)

        # exclude short subTS and modify dates of discos
        df, discos_mod = exclude_short_parts(df, discos, weeks_tresh_part=weeks_tresh_part, 
                                            weeks_tresh_all=weeks_tresh_all)
        if df is None:
            print('stanica {} bola preskocena pre prilis kratky TS'.format(station))
            continue

        # fit station by line
        # jumps in OLS model
        if fit_fullTS:
            df_fit, yearly_vels, yearly_vels_SDs = fit_fullTS_w_jumps(df, discos_mod, coors, outlier_trash=outlier_trash, 
                                        plot_outliers=plot_each_outliers, period_year=period_year, use_weights=use_weights)
            if plot_each_fit:
                plot_triple_coors(df_fit, coors, '{}'.format(station), draw=False, discos=discos_mod)
        # # fitted separatly and AVGed
        else:
            df_fit = fit_subTSs_separ(df, discos_mod, coors, outlier_trash=outlier_trash, period_year=period_year)
            if plot_each_fit:
                plot_triple_coors(df_fit, coors, '{}, separate fitting'.format(station), discos=discos_mod)
            # # get AVG slope
            yearly_vels = get_averaged_velo(df_fit, coors, station, plots=False)
            yearly_vels_SDs = None

        stations_velos[station] = yearly_vels
        stations_velos_SDs[station] = yearly_vels_SDs

        # get residual TS
        df_res = pd.DataFrame(index=df_fit.index)
        for coor in coors:
            fitcol = 'trend_' + coor
            df_res[coor] = df_fit[coor] - df_fit[fitcol]
        df_res['station'] = station
        stations_dfres[station] = df_res

        if plot_each_res:
            plot_triple_coors(df_res, coors, '{}, residuals'.format(station), plot_trends=False)

        # df_bern_velo = Bern_velos[Bern_velos['station'] == station]
        # print(df_bern_velo)

    return stations_velos, stations_velos_SDs, stations_dfres


def plot_velocities(COORS_FILE, SHP_FILE, stations_velos, stations_velos_SDs, ellipse=True):
    ''' Plot horizontal and vertical velocities'''
    df = get_velos_df(COORS_FILE, stations_velos, stations_velos_SDs)
    # colors based on direction (Hz velos)
    angles = np.arctan2(df['v_n'].values, df['v_e'].values)
    ang = pd.Series(angles)
    #3 colors
    # ang_labeled = pd.cut(ang, [-np.pi , -np.pi*2/3, np.pi/2, np.pi*3/4, np.pi], labels=[0,1,2,3])
    # ang_labeled.replace(3,0, inplace=True)
    # 2 colors
    ang_labeled = pd.cut(ang, [-np.pi , -np.pi*2/3, np.pi/2, np.pi*3/4, np.pi], labels=[0,1,2,3])
    ang_labeled.replace(3,0, inplace=True)
    ang_labeled.replace(2,1, inplace=True)

    # Horizontal velos test
    plot_SVK(SHP_FILE, draw=True)
    plt.title("Horizontal velocities")
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')

    # ax = plt.gca()
    # ax.set_xlim([16,22.8])
    # ax.set_ylim([47.3,50.6])
    
    if ellipse:
        # quiver_scale_hz = 0.003
        quiver_scale_hz = 0.002
        Q = plt.quiver(df['la'].values, df['fi'].values, df['v_e'].values, df['v_n'].values, ang_labeled,
                       pivot='tail', units='x', edgecolor='k', facecolor='r', linewidth=.5, width=0.03, 
                       alpha=1, scale=quiver_scale_hz, scale_units='x')

        # plot error ellipses
        ax = plt.gca()
        for i in range(len(df['la'])):
            la = df['la'].values[i]
            fi = df['fi'].values[i]
            v_e = df['v_e'].values[i]
            v_n = df['v_n'].values[i]
            Sv_n = df['Sv_n'].values[i]
            Sv_e = df['Sv_e'].values[i]
            plot_error_ellipse(ax, (la+v_e/quiver_scale_hz ,fi+v_n/quiver_scale_hz*WIDTH_ASPECT), Sv_n/quiver_scale_hz, Sv_e/quiver_scale_hz)
        
    else:
        Q = plt.quiver(df['la'].values, df['fi'].values, df['v_e'].values, df['v_n'].values, ang_labeled,
                   pivot='tail', units='inches', edgecolor='k', facecolor='r', linewidth=.5, alpha=1)

    qk = plt.quiverkey(Q, 0.8, 0.8, 0.002, r'2mm/year', labelpos='E',
                       coordinates='figure', color='r')

    plt.scatter(df['la'].values, df['fi'].values, c='black', marker='x')
    for idx, row in df.iterrows():
        plt.text(row['la'], row['fi'], str(row['station']), fontsize=12)

    # fig = plt.gcf()
    # fig.set_size_inches(15, 15, forward=True)
    # fig.savefig('test2png10.png')
    plt.draw()


    # Vertical velos
    # color identifiaction 
    ups = (df['v_u'] > 0).astype(int)
    # Vertical velos
    plot_SVK(SHP_FILE, draw=True)
    plt.title("Vertical velocities")
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    if ellipse:
        quiver_scale_up = 0.003
        Q = plt.quiver(df['la'].values, df['fi'].values, 0, df['v_u'].values, ups.values,
                       pivot='tail', units='x', linewidth=.5, width=0.03, 
                       alpha=1, scale=quiver_scale_up, scale_units='x')
    else:
        Q = plt.quiver(df['la'].values, df['fi'].values, 0, df['v_u'].values, ups.values,
                       pivot='tail', units='inches', linewidth=.5)

    qk = plt.quiverkey(Q, 0.8, 0.8, 0.002, r'2mm/year', labelpos='E',
                       coordinates='figure', color='r')

    plt.scatter(df['la'].values, df['fi'].values, c='black', marker='x')

    for idx, row in df.iterrows():
        plt.text(row['la'], row['fi'], str(row['station']), fontsize=12)

    # plot errors
    ax = plt.gca()
    # ax.set_xlim([16,22.8])
    # ax.set_ylim([47.3,50.6])
    for i in range(len(df['la'])):
        la = df['la'].values[i]
        fi = df['fi'].values[i]
        v_u = df['v_u'].values[i]
        Sv_u = df['Sv_u'].values[i]
        if ellipse:
            ax.errorbar(la, fi, yerr=Sv_u/quiver_scale_up, color='r')
    plt.show()




plt.rcParams['image.cmap'] = 'brg'
if __name__ == '__main__':
    DST_FILE = constants.DST_FILE
    EXCL_FILE = constants.EXCL_FILE
    TS_DIR = constants.TS_DIR
    COORS_FILE = constants.COORS_FILE
    SHP_FILE = constants.BORDERS_SHP

    stations_velos, SDs, stations_dfres = get_final_velocities(DST_FILE, EXCL_FILE, TS_DIR, plot_each_fit=False, 
                                    plot_each_outliers=False, period_year=True, stat=None, fit_fullTS=True, 
                                    outlier_trash=3, plot_each_res=False, use_weights=False)
    # save results (velos, SDs, ...)
    # df_velos = get_velos_df(COORS_FILE, stations_velos, SDs)
    # df_velos = df_velos.dropna()
    # df_velos = df_velos.round(8)
    # df_velos.to_csv('final_velos.csv', index=False)

    # save residuals
    # df_res_all = pd.concat(list(stations_dfres.values()))
    # df_res_all.to_csv('residuals.csv', float_format='%.8f')

    # # plot SKPOS velos
    # plot_velocities(COORS_FILE, SHP_FILE, stations_velos, SDs)
    # for sta, dat in stations_velos.items():
    #     print(sta, dat['n']*1000, dat['e']*1000, dat['u']*1000)

    # plot velos from EPN 
    # MOP2 musi mat daky bug vo vyske na EPN !!!
    EPN_velos = {'BBYS': {'n': -0.1, 'e': -0.5, 'u': -0.9},
                'BOR1': {'n': -0.2, 'e': -0.3, 'u': -1.2},
                'JOZE': {'n': -0.1, 'e': -0.1, 'u': -0.7},
                'SULP': {'n': 0.1, 'e':-0.6, 'u': -0.7},
                'CFRM': {'n': 0, 'e':-0.3,'u': 0.4},
                'TUBO': {'n': 0, 'e':-0.2, 'u': -0.2},
                'GOPE': {'n': -0.1, 'e':-0.2, 'u': -0.2},
                'GRAZ': {'n': 0.5, 'e':0.5, 'u': -0.8},
                'PENC': {'n': 0, 'e':0.5, 'u': -1.2},
                'OROS': {'n': 0, 'e':0, 'u': -1.7},
                'BAIA': {'n': -0.5, 'e':-0.3, 'u': -0.2},
                'USDL': {'n': 0, 'e':-0.2, 'u': -1.3},
                'UZHL': {'n': -0.5, 'e':-0.3, 'u': -0.5},
                'MOP2': {'n': 0.2, 'e':0.1, 'u': 0.2} }

    for st, coords in EPN_velos.copy().items():
        for coord, value in coords.items():
            EPN_velos[st][coord] = value / 1000

    EPN_SDs = {'BBYS': {'n': 0.02, 'e': 0.01, 'u': 0.08},
                'BOR1': {'n': 0.01, 'e': 0.01, 'u': 0.03},
                'JOZE': {'n': 0.01, 'e': 0.01, 'u': 0.04},
                'SULP': {'n': 0.02, 'e':0.01, 'u': 0.06},
                'CFRM': {'n': 0.04, 'e':0.04,'u': 0.14},
                'TUBO': {'n': 0.03, 'e':0.02, 'u': 0.09},
                'GOPE': {'n': 0.03, 'e':0.03, 'u': 0.08},
                'GRAZ': {'n': 0.02, 'e':0.02, 'u': 0.06},
                'PENC': {'n': 0.02, 'e':0.01, 'u': 0.06},
                'OROS': {'n': 0.04, 'e':0.03, 'u': 0.13},
                'BAIA': {'n': 0.02, 'e':0.02, 'u': 0.07},
                'USDL': {'n': 0.03, 'e':0.02, 'u': 0.10},
                'UZHL': {'n': 0.02, 'e':0.01, 'u': 0.08},
                'MOP2': {'n': 0.01, 'e':0.01, 'u': 0.04} }

    for st, coords in EPN_SDs.copy().items():
        for coord, value in coords.items():
            EPN_SDs[st][coord] = value / 1000

    plot_velocities(COORS_FILE, SHP_FILE, EPN_velos, EPN_SDs)

    


