#!/usr/local/bin/python

# This script provides the supporting functions for the plotting script.


import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from calendar import monthrange


# setting up the paths
datapath = '/home/jackson/Work/Data/IMERG-PCMK-MRMS/'
mrmspath = '/media/jackson/Vault/MRMS/Stage III/'
pcmkpath = '/media/jackson/Vault/GAG/PCMK/'
imergpath = '/media/jackson/Vault/IMERG-F/'
gprofpath = '/media/jackson/Vault/GPROF/SSMIS/'

satids = (0, 1, 3, 5, 7, 9)    # define the satellite IDs
satlab = ('IR only', 'IR+morph', 'morph only', 
          'TMI', 'AMSR', 'SSMIS', 'MHS', 'GMI')


def mask_days(data, snowdays, year0 = 2014, month0 = 4, day0 = 1, 
                  after = True):

    '''Mask an array of data with dims = (days, timestep) on days with snow, 
       as well as an option for days after snow (default: True).'''

    mask = np.zeros(data.shape, dtype = np.bool)

    for snowday in snowdays:
        index = (snowday - datetime(year0, month0, day0)).days
        if index >= 0 and index < len(data):
            mask[index] = 1
            if after:
                mask[index + 1] = 1

    return np.ma.masked_where(mask, data)


def read_imerg(year0, month0, year1, month1, path, additional = False):

    '''Read the IMERG data. If save files exist, read the files. If not, 
       compute from raw data and record the data into save files. Option to
       also read additional variables (HQprecipSource and IRkalmanFilterWeight) 
       exists (default: False).'''

    import h5py
    from glob import glob

    nday = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
            datetime(year0, month0, 1)).days + 1

    inpath = '%s%4d/%02d/%02d/' % (path, 2014, 4, 1)
    files = sorted(glob('%s3B-HHR*' % inpath))
    nt = len(files)

    with h5py.File(files[0]) as f:
        lats = f['Grid/lat'][:]
        lons = f['Grid/lon'][:]
        fillvalue = f['Grid/precipitationCal'].attrs['_FillValue']
        
    lat = np.where(lats ==  38.05)[0][0]
    lon = np.where(lons == -75.55)[0][0]

    imerg_file = '%simerg_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)
    ipsrc_file = '%sipsrc_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)
    irkal_file = '%sirkal_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)

    if os.path.exists(imerg_file) and additional == False:
        
        Pimerg = np.load(imerg_file)

    elif os.path.exists(imerg_file) and os.path.exists(ipsrc_file) and os.path.exists(irkal_file):
    
        Pimerg = np.load(imerg_file)
        Pipsrc = np.load(ipsrc_file)
        Pirkal = np.load(irkal_file)
        
    else:

        Pimerg = np.ma.masked_all([nday, nt])   # IMERG rain rate
        Pipsrc = np.ma.masked_all([nday, nt])   # IMERG precip source
        Pirkal = np.ma.masked_all([nday, nt])   # IMERG IR Kalman filter weight

        dor = 0   # day of record
        for year in range(year0, year1 + 1):

            if   year0 == year1: months = range(month0, month1 + 1)
            elif year  == year0: months = range(month0, 13)
            elif year  == year1: months = range(1, month1 + 1)
            else               : months = range(1, 13)

            for month in months:
                for day in range(1, monthrange(year, month)[1] + 1):
                    
                    inpath = '%s%4d/%02d/%02d/' % (path, year, month, day)
                    files = sorted(glob('%s3B-HHR*' % inpath))

                    for tt in range(nt):
                        with h5py.File(files[tt]) as f:
                            Pimerg[dor, tt] = f['Grid/precipitationCal'][lon, lat]
                            Pipsrc[dor, tt] = f['Grid/HQprecipSource'][lon, lat]
                            Pirkal[dor, tt] = f['Grid/IRkalmanFilterWeight'][lon, lat]

                    dor += 1

        Pimerg = np.ma.masked_values(Pimerg, fillvalue)
        Pipsrc = np.ma.masked_values(Pipsrc, fillvalue)
        Pirkal = np.ma.masked_values(Pirkal, fillvalue)

        with open(imerg_file, 'wb') as f:
            np.ma.dump(Pimerg, f)
        with open(ipsrc_file, 'wb') as f:
            np.ma.dump(Pipsrc, f)
        with open(irkal_file, 'wb') as f:
            np.ma.dump(Pirkal, f)

    if additional:
        return Pimerg, Pipsrc, Pirkal
    else:
        return Pimerg


def read_imerg_uncal(year0, month0, year1, month1, path, additional = False):

    '''Read the IMERG data. If save files exist, read the files. If not, 
       compute from raw data and record the data into save files. Option to
       also read additional variables (HQprecipSource and IRkalmanFilterWeight) 
       exists (default: False).'''

    import h5py
    from glob import glob

    nday = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
            datetime(year0, month0, 1)).days + 1

    inpath = '%s%4d/%02d/%02d/' % (path, 2014, 4, 1)
    files = sorted(glob('%s3B-HHR*' % inpath))
    nt = len(files)

    with h5py.File(files[0]) as f:
        lats = f['Grid/lat'][:]
        lons = f['Grid/lon'][:]
        fillvalue = f['Grid/precipitationCal'].attrs['_FillValue']
        
    lat = np.where(lats ==  38.05)[0][0]
    lon = np.where(lons == -75.55)[0][0]

    imerg_file = '%simergUncal_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)
    ipsrc_file = '%sipsrc_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)
    irkal_file = '%sirkal_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)

    if os.path.exists(imerg_file) and additional == False:
        
        Pimerg = np.load(imerg_file)

    elif os.path.exists(imerg_file) and os.path.exists(ipsrc_file) and os.path.exists(irkal_file):
    
        Pimerg = np.load(imerg_file)
        Pipsrc = np.load(ipsrc_file)
        Pirkal = np.load(irkal_file)
        
    else:

        Pimerg = np.ma.masked_all([nday, nt])   # IMERG rain rate
        Pipsrc = np.ma.masked_all([nday, nt])   # IMERG precip source
        Pirkal = np.ma.masked_all([nday, nt])   # IMERG IR Kalman filter weight

        dor = 0   # day of record
        for year in range(year0, year1 + 1):

            if   year0 == year1: months = range(month0, month1 + 1)
            elif year  == year0: months = range(month0, 13)
            elif year  == year1: months = range(1, month1 + 1)
            else               : months = range(1, 13)

            for month in months:
                for day in range(1, monthrange(year, month)[1] + 1):
                    
                    inpath = '%s%4d/%02d/%02d/' % (path, year, month, day)
                    files = sorted(glob('%s3B-HHR*' % inpath))

                    for tt in range(nt):
                        with h5py.File(files[tt]) as f:
                            Pimerg[dor, tt] = f['Grid/precipitationUncal'][lon, lat]
                            Pipsrc[dor, tt] = f['Grid/HQprecipSource'][lon, lat]
                            Pirkal[dor, tt] = f['Grid/IRkalmanFilterWeight'][lon, lat]

                    dor += 1

        Pimerg = np.ma.masked_values(Pimerg, fillvalue)
        Pipsrc = np.ma.masked_values(Pipsrc, fillvalue)
        Pirkal = np.ma.masked_values(Pirkal, fillvalue)

        with open(imerg_file, 'wb') as f:
            np.ma.dump(Pimerg, f)
        with open(ipsrc_file, 'wb') as f:
            np.ma.dump(Pipsrc, f)
        with open(irkal_file, 'wb') as f:
            np.ma.dump(Pirkal, f)

    if additional:
        return Pimerg, Pipsrc, Pirkal
    else:
        return Pimerg


def read_pcmk(year0, month0, year1, month1, path, 
              exclude = [5, 10, 12, 18, 19, 22, 25]):

    '''Read the PCMK data. The option 'exclude' allows the specification of 
       gauges to exclude (e.g. has been moved).'''

    nday = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
            datetime(year0, month0, 1)).days + 1

    gauges = [ii for ii in range(1, 26) if ii not in exclude]

    Ppcmk_raw = np.ma.zeros([len(gauges), 2, nday, 48])

    for gg, gauge in enumerate(gauges):
        for bu, bucket in enumerate(('A', 'B')):

            for year in range(year0, year1 + 1):

                if   year0 == year1: months = range(month0, month1 + 1)
                elif year  == year0: months = range(month0, 13)
                elif year  == year1: months = range(1, month1 + 1)
                else               : months = range(1, 13)

                # set day of record for first day of the year
                if year == year0:
                    sday = -(datetime(year0, month0, 1) - datetime(year0, 1, 1)).days
                else:
                    sday = (datetime(year, 1, 1) - datetime(year0, month0, 1)).days

                with open('%s%4d/IOWA-NASA00%02d_%s-%d.gag' % (path, year, gauge, 
                                                        bucket, year), 'r') as f:
                    data = [line.split() for line in f.readlines()]

                for ll in data[2:]:
                    if int(ll[1]) in months:   # if the month of the entry is in our range
                        dor = int(ll[3]) + sday - 1    # day of record
                        hr = int(ll[4])    # hour of the day
                        hod = int(ll[4]) * 2 + int(ll[5]) // 30    # half-hour of the day
                        Ppcmk_raw[gg, bu, dor, hod] += float(ll[7])

    Ppcmk_diff = np.diff(Ppcmk_raw, axis = 1).squeeze()
    Ppcmk_mean = np.ma.mean(Ppcmk_raw, 1)
    Ppcmk_maskpair = (np.abs(Ppcmk_diff) >= 0.5 * Ppcmk_mean) * (np.abs(Ppcmk_diff) > 0.3)
    Ppcmk = np.ma.mean(np.ma.masked_where(Ppcmk_maskpair, np.ma.mean(Ppcmk_raw, 1)), 0)

    return Ppcmk


def read_mrms(year0, month0, year1, month1, path, minrqi = 100):

    '''Read the MRMS data. If save files exist, read the files. If not, 
       compute from raw data and record the data into save files. Option to
       set minimum RQI (default = 100).'''

    import gzip

    nday = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
            datetime(year0, month0, 1)).days + 1

    mrms_file = '%smrms_%4d%02d-%4d%02d.npy' % (datapath, year0, month0, year1, month1)

    if os.path.exists(mrms_file):
        
        Pmrms = np.load(mrms_file)
        
    else:

        Pmrms = np.ma.masked_all([nday, 24, 10, 10])
        nmissing = 0

        dor = 0   # day of record
        
        for year in range(year0, year1 + 1):

            if   year0 == year1: months = range(month0, month1 + 1)
            elif year  == year0: months = range(month0, 13)
            elif year  == year1: months = range(1, month1 + 1)
            else               : months = range(1, 13)

            for month in months:
                for day in range(1, monthrange(year, month)[1] + 1):
                    
                    for hour in range(24):

                        ts = datetime(year, month, day, hour) + timedelta(hours = 1)
                        yy, mm, dd, hh = ts.year, ts.month, ts.day, ts.hour
                        filename1 = '1HGCF.%4d%02d%02d.%04d00.asc.gz' % (yy, mm, dd, hh * 100)
                        filename2 = '1HRQI.%4d%02d%02d.%04d00.asc.gz' % (yy, mm, dd, hh * 100)

                        try:
                            with gzip.open('%s%4d/%02d/%s' % (path, yy, mm, filename1), 'rb') as f:
                                data = np.array([[float(ii) for ii in jj.split()[5440 : 5450]] 
                                                  for jj in f.readlines()[6 + 1690 : 6 + 1700]])
                            with gzip.open('%s%4d/%02d/%s' % (path, yy, mm, filename2), 'rb') as f:
                                rqi = np.array([[float(ii) for ii in jj.split()[5440 : 5450]] 
                                                 for jj in f.readlines()[6 + 1690 : 6 + 1700]])

                                Pmrms[dor, hour] = np.ma.masked_where(rqi < minrqi, 
                                                       np.ma.masked_values(data, -999.))

                        except FileNotFoundError:
                            nmissing += 1

                    dor += 1

        with open(mrms_file, 'wb') as f:
            np.ma.dump(Pmrms, f)

    return Pmrms


def accum(data_base, time, base = 1, rate = False):

    '''Function to calculate accumulation for longer periods from the base accumulation
       (default: base = 1 hr). If rate is True, then values returned are rates and not
       accumulation (i.e. in units of mm / hr and not mm).
       Note: data_base must be in dimensions of (day, base) and in units of mm / base.'''

    if (24 / time) % 1:

        print('Error: specified period not a factor of 24.')
        return False

    else:

        step = time / base    # no. of steps to average over

        x = [np.ma.sum(data_base[:, ii * step : (ii + 1) * step], 1)
             for ii in range(24 // time)]

        if rate:
            return np.ma.array(x).T / step
        else:
            return np.ma.array(x).T


def calc_error(x, y, error = 'total', norm = 'ref', threshold = 0.1):

    '''Calculate the error of two arrays for values above the threshold.
       Important: x is the truth while y is the variable. Choice of error 
       type ('total', 'abs', 'rmse') and normalization ('none', 'size', 
       'ref'). Default threshold = 0.1 with boolean OR.'''

    x = x.flatten()
    y = y.flatten()

    th = np.array((x >= threshold) * (y >= threshold), dtype = np.bool)

    if   norm == 'none':
        N = 1
    elif norm == 'size':
        N = np.ma.sum(th)
    elif norm == 'ref':
        N = np.ma.sum(x[th])
    else:
        print('Error: unknown type of normalization.')
    
    if   error == 'total':
        err = np.ma.sum(y[th] - x[th])
    elif error == 'abs':
        err = np.ma.sum(np.ma.abs(y[th] - x[th]))
    elif error == 'rmse':
        err = np.ma.sqrt(np.ma.sum(th) * np.ma.sum((y[th] - x[th]) ** 2))
    else:
        print('Error: unknown type of error.')

    return err / N


def read_data(year0, month0, year1, month1):

    Pimerg = read_imerg(year0, month0, year1, month1, path = imergpath)
    Ppcmk = read_pcmk(year0, month0, year1, month1, path = pcmkpath)
    Pmrms = read_mrms(year0, month0, year1, month1, path = mrmspath)

    snowdays = []
    with open(datapath + 'snowdays', 'r') as f:
        for day in f.readlines():
            year, month, day = day.split()
            snowdays.append(datetime(int(year), int(month), int(day)))

    nday = (datetime(2015, 10, 7) - datetime(2015, 9, 21)).days + 1
    missing = [(datetime(2015, 9, 21) + timedelta(days = day))
               for day in range(nday)]

    Pimerg = mask_days(mask_days(Pimerg, snowdays), missing, after = False)
    Ppcmk = mask_days(mask_days(Ppcmk, snowdays), missing, after = False)
    Pmrms = mask_days(mask_days(Pmrms, snowdays), missing, after = False)
    Pmrms1 = np.ma.mean(np.ma.mean(Pmrms, -1), -1)    # over IMERG grid
    Pmrms2 = np.ma.mean(np.ma.mean(Pmrms[:, :, :5, :6], -1), -1) # over PCMK grid

    P = {}

    P['imerg_1'] = accum(Pimerg, 1, base = 0.5) / 2
    P['pcmk_1'] = accum(Ppcmk, 1, base = 0.5)
    P['mrms1_1'] = Pmrms1.copy()
    P['mrms2_1'] = Pmrms2.copy()

    return P


def read_data_uncal(year0, month0, year1, month1):

    Pimerg = read_imerg_uncal(year0, month0, year1, month1, path = imergpath)
    Ppcmk = read_pcmk(year0, month0, year1, month1, path = pcmkpath)
    Pmrms = read_mrms(year0, month0, year1, month1, path = mrmspath)

    snowdays = []
    with open(datapath + 'snowdays', 'r') as f:
        for day in f.readlines():
            year, month, day = day.split()
            snowdays.append(datetime(int(year), int(month), int(day)))

    nday = (datetime(2015, 10, 7) - datetime(2015, 9, 21)).days + 1
    missing = [(datetime(2015, 9, 21) + timedelta(days = day))
               for day in range(nday)]

    Pimerg = mask_days(mask_days(Pimerg, snowdays), missing, after = False)
    Ppcmk = mask_days(mask_days(Ppcmk, snowdays), missing, after = False)
    Pmrms = mask_days(mask_days(Pmrms, snowdays), missing, after = False)
    Pmrms1 = np.ma.mean(np.ma.mean(Pmrms, -1), -1)    # over IMERG grid
    Pmrms2 = np.ma.mean(np.ma.mean(Pmrms[:, :, :5, :6], -1), -1) # over PCMK grid

    P = {}

    P['imerg_1'] = accum(Pimerg, 1, base = 0.5) / 2
    P['pcmk_1'] = accum(Ppcmk, 1, base = 0.5)
    P['mrms1_1'] = Pmrms1.copy()
    P['mrms2_1'] = Pmrms2.copy()

    return P


def read_platform_data(year0, month0, year1, month1):

    snowdays = []
    with open(datapath + 'snowdays', 'r') as f:
        for day in f.readlines():
            year, month, day = day.split()
            snowdays.append(datetime(int(year), int(month), int(day)))

    # add days of missing data into "snowdays"
    nday = (datetime(2015, 10, 7) - datetime(2015, 9, 21)).days + 1
    missing = [(datetime(2015, 9, 21) + timedelta(days = day))
               for day in range(nday)]

    Pimerg, Pipsrc, Pirkal = read_imerg(year0, month0, year1, month1, 
        path = imergpath, additional = True)
    Pimerg = mask_days(mask_days(Pimerg, snowdays), missing, after = False)
    Pipsrc = mask_days(mask_days(Pipsrc, snowdays), missing, after = False)
    Pirkal = mask_days(mask_days(Pirkal, snowdays), missing, after = False)

    Ppcmk = read_pcmk(year0, month0, year1, month1, path = pcmkpath) * 2
    Ppcmk = mask_days(mask_days(Ppcmk, snowdays), missing, after = False)

    # sort by satellite

    Pimerg_flat = Pimerg.flatten()
    Pipsrc_flat = Pipsrc.flatten()
    Pirkal_flat = Pirkal.flatten()
    Ppcmk_flat = Ppcmk.flatten()

    imerg_satids = [[] for ii in range(len(satlab))]
    pcmk_satids = [[] for ii in range(len(satlab))]
    anom_satids = [[] for ii in range(len(satlab))]

    for ii in range(len(Pimerg_flat)):
        if not np.ma.is_masked(Pipsrc_flat[ii]):
            if   Pirkal_flat[ii] == 100:
                imerg_satids[0].append(Pimerg_flat[ii])
                pcmk_satids[0].append(Ppcmk_flat[ii])
                anom_satids[0].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
            elif Pipsrc_flat[ii] == 0:
                if   Pirkal_flat[ii] == 0:
                    imerg_satids[2].append(Pimerg_flat[ii])
                    pcmk_satids[2].append(Ppcmk_flat[ii])
                    anom_satids[2].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
                else:
                    imerg_satids[1].append(Pimerg_flat[ii])
                    pcmk_satids[1].append(Ppcmk_flat[ii])
                    anom_satids[1].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
            else:
                value1 = Pimerg_flat[ii]
                value2 = Ppcmk_flat[ii]
                value3 = Pimerg_flat[ii] - Ppcmk_flat[ii]
                imerg_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value1)
                pcmk_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value2)
                anom_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value3)

    return imerg_satids, pcmk_satids, anom_satids


def read_platform_data_uncal(year0, month0, year1, month1):

    snowdays = []
    with open(datapath + 'snowdays', 'r') as f:
        for day in f.readlines():
            year, month, day = day.split()
            snowdays.append(datetime(int(year), int(month), int(day)))

    # add days of missing data into "snowdays"
    nday = (datetime(2015, 10, 7) - datetime(2015, 9, 21)).days + 1
    missing = [(datetime(2015, 9, 21) + timedelta(days = day))
               for day in range(nday)]

    Pimerg, Pipsrc, Pirkal = read_imerg_uncal(year0, month0, year1, month1, 
        path = imergpath, additional = True)
    Pimerg = mask_days(mask_days(Pimerg, snowdays), missing, after = False)
    Pipsrc = mask_days(mask_days(Pipsrc, snowdays), missing, after = False)
    Pirkal = mask_days(mask_days(Pirkal, snowdays), missing, after = False)

    Ppcmk = read_pcmk(year0, month0, year1, month1, path = pcmkpath) * 2
    Ppcmk = mask_days(mask_days(Ppcmk, snowdays), missing, after = False)

    # sort by satellite

    Pimerg_flat = Pimerg.flatten()
    Pipsrc_flat = Pipsrc.flatten()
    Pirkal_flat = Pirkal.flatten()
    Ppcmk_flat = Ppcmk.flatten()

    imerg_satids = [[] for ii in range(len(satlab))]
    pcmk_satids = [[] for ii in range(len(satlab))]
    anom_satids = [[] for ii in range(len(satlab))]

    for ii in range(len(Pimerg_flat)):
        if not np.ma.is_masked(Pipsrc_flat[ii]):
            if   Pirkal_flat[ii] == 100:
                imerg_satids[0].append(Pimerg_flat[ii])
                pcmk_satids[0].append(Ppcmk_flat[ii])
                anom_satids[0].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
            elif Pipsrc_flat[ii] == 0:
                if   Pirkal_flat[ii] == 0:
                    imerg_satids[2].append(Pimerg_flat[ii])
                    pcmk_satids[2].append(Ppcmk_flat[ii])
                    anom_satids[2].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
                else:
                    imerg_satids[1].append(Pimerg_flat[ii])
                    pcmk_satids[1].append(Ppcmk_flat[ii])
                    anom_satids[1].append(Pimerg_flat[ii] - Ppcmk_flat[ii])
            else:
                value1 = Pimerg_flat[ii]
                value2 = Ppcmk_flat[ii]
                value3 = Pimerg_flat[ii] - Ppcmk_flat[ii]
                imerg_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value1)
                pcmk_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value2)
                anom_satids[satids.index(Pipsrc_flat[ii]) + 2].append(value3)

    return imerg_satids, pcmk_satids, anom_satids


def scatter_log(x, y, size = 3, threshold = 0.1):

    x = x.flatten()
    y = y.flatten()

    th = np.ma.filled((x >= threshold) * (y >= threshold), False)

    plt.scatter(x[th], y[th], s = size, color = 'k', edgecolor = 'none')
    plt.plot([0.1, 100], [0.1, 100], color = '0.5', ls = '--')
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.axis([0.1, 100, 0.1, 100])
    plt.text(0.96, 0.03, 'n = %3d' % np.sum(th), ha = 'right', va = 'bottom', 
             transform = plt.gca().transAxes)
    plt.grid()

    return None


def compare(x, mode, minrain):
    if   mode == 'N': return x.flatten() <  minrain
    elif mode == 'Y': return x.flatten() >= minrain


def fit_mem(x, y, minrain):
    '''Performs fit to the multiplicative error model. Important: x is the 
       truth while y is the variable.'''

    import statsmodels.api as sm

    th = np.ma.filled((x >= minrain) * (y >= minrain), False)
    
    Y = np.log(y[th])
    X = sm.add_constant(np.log(x[th]))
    
    results = sm.OLS(Y, X).fit()
    
    try:
        alpha, beta = results.params
        sigma = np.std(results.resid)
    except ValueError:
        alpha, beta, sigma = np.nan, np.nan, np.nan

    return alpha, beta, sigma


#def fit_mem(x, y, minrain):

#    '''Using the Orthogonal Distance Regression.'''

#    from scipy import odr

#    th = np.ma.filled((x >= minrain) * (y >= minrain), False)
#    X = np.log(x[th])
#    Y = np.log(y[th])

#    def f(B, x):
#        return B[0] + B[1] * x

#    linear = odr.Model(f)
#    mydata = odr.Data(X, Y)
#    myodr = odr.ODR(mydata, linear, beta0 = [1., 0])
#    myoutput = myodr.run()

#    return myoutput.beta[0], myoutput.beta[1], np.sqrt(myoutput.res_var)



def calc_corr(x, y, minrain):
    th = np.array((x >= minrain) * (y >= minrain), dtype = np.bool)
    if np.sum(th) < 2:
        return np.nan
    else:
        return np.ma.corrcoef(x[th], y[th])[0, 1]
