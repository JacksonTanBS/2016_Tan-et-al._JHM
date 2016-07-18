#!/usr/local/bin/python

# This script plots the diagrams in this paper.


#--- PRELIMINARIES ---#


import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import sys
from datetime import datetime
from funcs import *

option = sys.argv[1:]

# setting up the parameters
year0, year1 = 2014, 2015
month0, month1 = 4, 9
nday = (datetime(year1, month1, monthrange(year1, month1)[1]) - 
        datetime(year0, month0, 1)).days + 1
minrain = 0.1                  # minimum rain threshold
periods = (1, 3, 6, 12, 24)    # accumulation periods to calculate
datalab = {'imerg': '$P_\mathrm{I}$', 'pcmk': '$P_\mathrm{G}$', 
           'mrms1': '$P_\mathrm{MI}$', 'mrms2': '$P_\mathrm{MG}$'}
subplab = ('(a)', '(b)', '(c)', '(d)')
satids = (0, 1, 3, 5, 7, 9)    # define the satellite IDs
satlab = ('IR only', 'IR+morph', 'morph only', 
          'TMI', 'AMSR', 'SSMIS', 'MHS', 'GMI')

# setting up the paths
datapath = '/home/jackson/Work/Data/IMERG-PCMK-MRMS/'
mrmspath = '/media/jackson/Vault/MRMS/Stage III/'
pcmkpath = '/media/jackson/Vault/GAG/PCMK/'
imergpath = '/media/jackson/Vault/IMERG-F/'
gprofpath = '/media/jackson/Vault/GPROF/'

# plot configurations
scol = 3.503    # single column (89 mm)
dcol = 7.204    # double column (183 mm)
flpg = 9.724    # full page length (247 mm)
plt.rcParams['font.size'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.sans-serif'] = ['TeX Gyre Heros', 'Helvetica',
                                   'Bitstream Vera Sans']
plt.rcParams['pdf.fonttype'] = 42
cols = brewer2mpl.get_map('Set1', 'Qualitative', '4').hex_colors


#--- DIAGRAMS ---#


if 'map' in option:

    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Rectangle

    mindist = 0.001
    gauges = list(range(1, 26))
    gauges_coord = np.ma.masked_all([len(gauges), 2])

    def check_gauge_position():
        for bu, bucket in enumerate(('A', 'B')):
            for year in range(year0, year1 + 1):

                if   year0 == year1: months = range(month0, month1 + 1)
                elif year  == year0: months = range(month0, 13)
                elif year  == year1: months = range(1, month1 + 1)
                else               : months = range(1, 13)

                filename = 'NASA00%02d_%s_%d.gag' % (gauge, bucket, year)
                with open('%s%4d/%s' % (pcmkpath, year, filename), 'r') as f:
                    data = [line.split() for line in f.readlines()]

                lats = []
                lons = []

                for ll in data[2:]:
                    if int(ll[1]) in months:   # if the month of entry in range
                        lat, lon = float(ll[8]), float(ll[9])

                        lats.append(lat)
                        lons.append(lon)

                        if (np.abs(np.mean(lons) - lon) > mindist or 
                            np.abs(np.mean(lats) - lat) > mindist):
                            gauges_moved.append(gauge)

                            return None, None

        return np.mean(lats), np.mean(lons)

    gauges_moved = []

    for gg, gauge in enumerate(gauges):
        gauges_coord[gg] = check_gauge_position()

    gauges_fixed = [ii for ii in gauges if ii not in gauges_moved]

    osm = plt.imread('%s/PCMK_osm_alt.png' % pcmkpath)

    plt.figure(figsize = (dcol, 0.75 * scol))

    # US map
    ax1 = plt.axes([0.1, 0.1, 0.375, 0.8])
    m = Basemap(projection = 'cyl', llcrnrlat = 20, urcrnrlat = 60, 
                llcrnrlon = -130, urcrnrlon = -70, lat_ts = 40, 
                resolution = 'i')
    m.drawcountries(linewidth = 0.5, color = 'k', zorder = 1)
    m.drawstates(linewidth = 0.5, color = '0.5', zorder = 1)
    m.drawcoastlines(linewidth = 0.5, zorder = 1)
    m.shadedrelief(scale = 1)
    ax1.add_patch(Rectangle((-76.5, 37), 1.5, 3, facecolor = 'none', 
                  linewidth = 1, edgecolor = 'r', zorder = 2))
    m.drawparallels(np.arange(20, 61, 20), linewidth = 0, labels = [1, 0, 0, 0])
    m.drawmeridians(np.arange(-130, -69, 20), linewidth = 0, 
                    labels = [0, 0, 0, 1])

    # Eastern Shore map
    ax2 = plt.axes([0.525, 0.1, 0.125, 0.8])
    m = Basemap(projection = 'cyl', llcrnrlat = 37, urcrnrlat = 40, 
                llcrnrlon = -76.5, urcrnrlon = -75, lat_ts = 0, 
                resolution = 'h')
    m.fillcontinents(color = '#759d92', zorder = 0)
    m.drawmapboundary(fill_color = '#d3e4f4', zorder = 0)
    m.drawstates(linewidth = 1, color = '0.5', zorder = 1)
    m.drawcoastlines(linewidth = 1, zorder = 1)
    ax2.add_patch(Rectangle((-75.6, 38), 0.1, 0.1, facecolor = 'none', 
                  linewidth = 1, edgecolor = 'indigo', zorder = 2))
    for spine in ('bottom', 'top', 'left', 'right'):
        ax2.spines[spine].set_color('r')

    # gauge location map
    ax3 = plt.axes([0.7, 0.1, 0.2, 0.8])
    for spine in ('bottom', 'top', 'left', 'right'):
        ax3.spines[spine].set_color('indigo')
    ax3.imshow(osm, extent = (-75.6, -75.5, 38.0, 38.1), 
               aspect = osm.shape[0] / osm.shape[1])
    ax3.scatter(gauges_coord[[ii - 1 for ii in gauges_fixed]][:, 1], 
                gauges_coord[[ii - 1 for ii in gauges_fixed]][:, 0], 
                c = 'g', s = 20, lw = 1, marker = 'x', zorder = 1)
    ax3.add_patch(Rectangle((-75.5994, 38.05), 0.06, 0.0493,
                        facecolor = 'none', edgecolor = 'g', lw = 1,
                        zorder = 2, alpha = 0.75))
    ax3.axis([-75.6, -75.5, 38.0, 38.1])
    ax3.set_xticks((-75.6, -75.54, -75.5))
    ax3.set_xticklabels(('75.60°W', '75.54°W', '75.50°W'))
    ax3.set_yticks((38.0, 38.05, 38.1))
    ax3.set_yticklabels(('38.00°N', '38.05°N', '38.10°N'))
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    plt.savefig('fig_map.pdf', dpi = 150)
    plt.close()


if 'all_contab' in option:

    P = read_data(year0, month0, year1, month1)

    pairs = (('mrms2', 'mrms1'), ('pcmk', 'mrms2'), 
             ('pcmk', 'imerg'))

    contabs = np.array([[[np.sum(compare(P['%s_1' % (y,)], ii, minrain) * 
                                 compare(P['%s_1' % (x,)], jj, minrain)) 
                          for ii in ('Y', 'N')] for jj in ('Y', 'N')] 
                          for x, y in pairs])
    celltext = [[['{:4d} ({:4.1f}%)'.format(contab[ii, jj], 
                  contab[ii, jj] / np.sum(contab) * 100) 
                  for jj in range(2)] for ii in range(2)] for contab in contabs]

    plt.figure(figsize = (scol, scol))
    for n, (x, y) in enumerate(pairs):
        ax = plt.subplot(411 + n)
        ax.axis('off')
        ax.text(-0.4, 0.93, subplab[n], ha = 'left', va = 'top', 
            transform = plt.gca().transAxes)
        rows = ('%s ≥ %3.1f mm / h' % (datalab[x], minrain), 
                '%s < %3.1f mm / h' % (datalab[x], minrain))
        columns = ('%s ≥ %3.1f mm / h' % (datalab[y], minrain), 
                   '%s < %3.1f mm / h' % (datalab[y], minrain))
        plt.table(cellText = celltext[n], rowLabels = rows, colLabels = columns,
                  loc = 'center')
    plt.savefig('tab_all_contab.pdf')
    plt.close()


if 'all_error' in option:

    P = read_data(year0, month0, year1, month1)

    pairs = (('mrms2', 'mrms1'), ('pcmk', 'mrms2'), 
             ('pcmk', 'imerg'))

    # calculate NTE, NAE
    errors = [[calc_error(P['%s_1' % (x,)], P['%s_1' % (y,)], error = error, 
                          norm = 'ref', threshold = minrain)
               for error in ('total', 'abs')] for x, y in pairs]

    # calculate correlation coefficient
    corr = [calc_corr(P['%s_1' % (x,)], P['%s_1' % (y,)], minrain) 
            for x, y in pairs]

    # calculate MEM parameters
    mem = [fit_mem(P['%s_1' % (x,)], P['%s_1' % (y,)], minrain) 
           for x, y in pairs]

    data = np.concatenate([np.around(errors, 3), 
                           np.around(corr, 3)[:, None],
                           np.around(mem, 3)], 1)
    celltext = [['%5.3f' % ii for ii in jj] for jj in data]

    columns = ('NME', 'NMAE', 'CC', r'$\alpha$', r'$\beta$', r'$\sigma$')
    rows = ['%s  %s, %s' % (subplab[n], datalab[y], datalab[x]) 
            for n, (x, y) in enumerate(pairs)]

    plt.figure(figsize = (1.1 * scol, 0.3 * scol))
    ax = plt.subplot(111)
    ax.axis('off')
    plt.table(cellText = celltext, rowLabels = rows, colLabels = columns,
              loc = 'center')
    plt.savefig('tab_all_error.pdf')
    plt.close()


if 'all_scatter' in option:

    P = read_data(year0, month0, year1, month1)

    pairs = (('mrms2', 'mrms1'), ('pcmk', 'mrms2'), 
             ('pcmk', 'imerg'))

    t = np.logspace(-1, 2, 101)

    plt.figure(figsize = (dcol, 0.5 * scol))
    plt.subplots_adjust(wspace = 0.4)
    for n, (x, y) in enumerate(pairs):
        plt.subplot(131 + n)
        scatter_log(P['%s_1' % x], P['%s_1' % y], size = 3,
                    threshold = minrain)
        alpha, beta, sigma = fit_mem(P['%s_1' % x], P['%s_1' % y], minrain)
        plt.plot(t, np.exp(alpha) * t ** beta, 'k')
        plt.xlabel('%s (mm / h)' % datalab[x])
        plt.ylabel('%s (mm / h)' % datalab[y])
        plt.text(0.03, 0.97, subplab[n], ha = 'left', va = 'top', 
            transform = plt.gca().transAxes)
    plt.savefig('fig_all_scatter.pdf')
    plt.close()


if 'all_metric' in option:

    P = read_data(year0, month0, year1, month1)

    pairs = (('mrms2', 'mrms1'), ('pcmk', 'mrms2'), 
             ('pcmk', 'imerg'))

    contabs = np.array([[[np.sum(compare(P['%s_1' % (x,)], ii, minrain) * 
                                 compare(P['%s_1' % (y,)], jj, minrain)) 
                          for ii in ('Y', 'N')] for jj in ('Y', 'N')] 
                          for x, y in pairs])

    pod = lambda x : (x[0, 0] / (x[0, 0] + x[1, 0]))
    far = lambda x : (x[0, 1] / (x[0, 0] + x[0, 1]))
    bias = lambda x : ((x[0, 0] + x[0, 1]) / (x[0, 0] + x[1, 0]))
    #csi = lambda x : (x[0, 0] / (x[0, 0] + x[0, 1] + x[1, 0]))
    def hss(x):
        N = np.sum(x)
        exp = 1 / N * ((x[0, 0] + x[1, 0]) * (x[0, 0] + x[0, 1]) + 
                       (x[1, 1] + x[1, 0]) * (x[1, 1] + x[0, 1]))
        return ((x[0, 0] + x[1, 1] - exp) / (N - exp))

    metrics = [['%5.3f' % f(contab) for f in (pod, far, bias, hss)] 
                for contab in contabs]

    columns = ('POD', 'FAR', 'BID', 'HSS')
    rows = ['%s  %s, %s' % (subplab[n], datalab[y], datalab[x]) 
            for n, (x, y) in enumerate(pairs)]

    plt.figure(figsize = (scol, 0.3 * scol))
    ax = plt.subplot(111)
    ax.axis('off')
    plt.table(cellText = metrics, rowLabels = rows, colLabels = columns,
              loc = 'center')
    plt.savefig('tab_all_metric.pdf')
    plt.close()


if 'sat_contab' in option:

    sat_imerg, sat_pcmk, _ = read_platform_data(year0, month0, year1, month1)

    hit, miss, false, correj = [], [], [], []

    for ss in range(len(satlab)):
        hit.append(np.sum(compare(np.array(sat_pcmk[ss]), 'Y', minrain) * 
                          compare(np.array(sat_imerg[ss]), 'Y', minrain)))
        miss.append(np.sum(compare(np.array(sat_pcmk[ss]), 'Y', minrain) * 
                           compare(np.array(sat_imerg[ss]), 'N', minrain)))
        false.append(np.sum(compare(np.array(sat_pcmk[ss]), 'N', minrain) * 
                            compare(np.array(sat_imerg[ss]), 'Y', minrain)))
        correj.append(np.sum(compare(np.array(sat_pcmk[ss]), 'N', minrain) * 
                             compare(np.array(sat_imerg[ss]), 'N', minrain)))

    def hss(H, M, F, C):
        N = H + M + F + C
        exp = 1 / N * ((H + M) * (H + F) + (C + M) * (C + F))
        return ((H + C - exp) / (N - exp))

    print('HSS for IR only: %5.3f' % hss(hit[0], miss[0], false[0], correj[0]))

    counts = np.vstack([hit, miss, false, correj])
    rawfrac = (counts / np.sum(counts, 0)) * 100

    barmax = 30
    frac = rawfrac.copy()
    frac[3] = frac[3] - (100 - barmax)

    contablab = ('hit', 'miss', 'false alarm', 'correct negative')

    y = np.arange(len(satlab))

    plt.figure(figsize = (0.75 * dcol, 0.5 * scol))
    for ii in range(4):
        plt.barh(y, frac[ii], color = cols[ii], left = np.sum(frac[0 : ii], 0),
                align = 'center', label = contablab[ii], edgecolor = 'none')
    plt.yticks(y)
    plt.setp(plt.gca(), 'yticklabels', satlab)
    plt.xlabel('(%)')
    plt.axis([0, barmax + 2, -0.4, len(satlab) - 1 + 0.4])
    plt.legend(loc = 1)
    for cc, count in enumerate(counts.T):
        polygon = plt.Polygon(([barmax + 2, cc], [barmax - 0.1, cc - 0.4], 
                               [barmax - 0.1, cc + 0.4]),
                              edgecolor = 'none', facecolor = cols[3])
        plt.gca().add_patch(polygon)
        plt.text(plt.axis()[1] + 3.0, cc, '{0:5d}'.format(np.sum(count)),
                 ha = 'right', va = 'center')
        plt.text(plt.axis()[1] + 6.7, cc, '({0:4.1f}%)'.format(np.sum(count) / 
                 np.sum(counts) * 100), ha = 'right', va = 'center')
        for ii in range(4):
            plt.text(np.sum(frac[:ii, cc]) + frac[ii, cc] / 2, cc,
                     '%4.1f' % rawfrac[ii, cc], color = 'w', fontsize = 7,
                     ha = 'center', va = 'center')

    plt.savefig('fig_sat_contab.pdf')
    plt.close()

    #np.savetxt('fig_sat_contab.txt', rawfrac.T, fmt = '%4.1f')


if 'sat_scatter' in option:

    sat_imerg, sat_pcmk, _ = read_platform_data(year0, month0, year1, month1)

    t = np.logspace(-1, 2, 101)

    plt.figure(figsize = (dcol, dcol))
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)

    for ss in range(len(satlab)):

        plt.subplot(331 + ss)
        scatter_log(np.array(sat_pcmk[ss]), np.array(sat_imerg[ss]), 
                    size = 5, threshold = minrain)

        if ss:    # skip IR because of only one point
            alpha, beta, sigma = fit_mem(np.array(sat_pcmk[ss]), 
                                         np.array(sat_imerg[ss]),
                                         minrain)
            plt.plot(t, np.exp(alpha) * t ** beta, 'k')

        plt.text(0.03, 0.97, satlab[ss], ha = 'left', va = 'top', 
            transform = plt.gca().transAxes)

        if ss > 4: plt.xlabel('%s (mm / h)' % datalab['pcmk'])
        if not ss % 3: plt.ylabel('%s (mm / h)' % datalab['imerg'])

        if   ss < 3:
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position("top")
        elif ss < 6:
            plt.gca().set_xticklabels([])
            plt.xlabel('')

        if   not (ss - 2) % 3:
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
        elif not (ss - 1) % 3:
            plt.gca().set_yticklabels([])
            plt.ylabel('')

    plt.savefig('fig_sat_scatter.pdf')
    plt.close()


if 'sat_error' in option:

    sat_imerg, sat_pcmk, _ = read_platform_data(year0, month0, year1, month1)

    # calculate NTE, NAE
    errors = [[calc_error(np.array(sat_pcmk[ss]), np.array(sat_imerg[ss]), 
                          error = error, norm = 'ref', threshold = minrain)
               for error in ('total', 'abs')] for ss in range(len(satlab))]

    # calculate correlation coefficient
    corr = [calc_corr(np.array(sat_pcmk[ss]), np.array(sat_imerg[ss]), 
            minrain) for ss in range(len(satlab))]

    # calculate MEM parameters
    mem = [fit_mem(np.array(sat_pcmk[ss]), np.array(sat_imerg[ss]), 
                   minrain) for ss in range(len(satlab))]

    size = [np.sum((np.array(sat_imerg[ss]) > minrain) * 
                   (np.array(sat_pcmk[ss]) > minrain))
            for ss in range(len(satlab))]

    data = np.concatenate([np.array(size)[:, None], 
                           np.around(errors, 3), 
                           np.around(corr, 3)[:, None],
                           np.around(np.array(mem), 3)], 1)

    # format table into appropriate decimal places
    tabletext = []
    for row in data:
        rowtext = []
        for cc, col in enumerate(row):
            if   cc == 0:
                rowtext.append('%3d' % col)
            elif np.isnan(col):
                rowtext.append('-')
            else:
                rowtext.append('%5.3f' % col)
        tabletext.append(rowtext)

    columns = ('n', 'NME', 'NMAE', 'CC', r'$\alpha$', r'$\beta$', r'$\sigma$')
    rows = satlab

    #plt.figure(figsize = (1.25 * scol, 0.5 * scol))
    plt.figure(figsize = (scol, 0.6 * scol))
    ax = plt.subplot(111)
    ax.axis('off')
    plt.table(cellText = tabletext, rowLabels = rows, colLabels = columns,
              loc = 'center')
    plt.savefig('tab_sat_error.pdf')
    plt.close()


if 'sat_violin' in option:

    from statsmodels.graphics.boxplots import violinplot
    from matplotlib.patches import Ellipse

    sat_imerg, sat_pcmk, sat_anom = read_platform_data(year0, month0, 
                                                       year1, month1)

    diff_satids = []
    for ss in range(len(satlab)):
        diff_satid = []
        for ii in range(len(sat_imerg[ss])):
            if sat_imerg[ss][ii] >= minrain or sat_pcmk[ss][ii] >= minrain:
                diff_satid.append(sat_anom[ss][ii])
                if ss == 5 and sat_anom[ss][ii] < -50:
                    pcmk_ssmis_min = sat_pcmk[ss][ii]  # PCMK for that event
        diff_satids.append(diff_satid)

    print('PCMK: %5.1f' % pcmk_ssmis_min)

    plt.figure(figsize = (scol, 0.75 * scol))

    violinplot(diff_satids, ax = plt.gca(), positions = range(len(sat_anom)), 
               show_boxplot = False,  plot_opts = {'violin_fc': '0.2', 
               'violin_ec': 'k', 'cutoff': True})

    bp = plt.boxplot(diff_satids, positions = range(len(sat_anom)), 
                whis = [10, 90])
    for flier in bp['fliers']:
        flier.set(marker = 'x', mec = 'k', ms = 4, alpha = 1)
    plt.ylim([-54, 32])

    circle = Ellipse((5, min(diff_satids[5])), 0.5, 6.3, color = 'r', 
                     fill = False)
    plt.gca().add_artist(circle)

    plt.xticks(range(len(diff_satids)), satlab, rotation = 'vertical')
    plt.setp(plt.gca(), 'xticklabels', satlab)
    plt.ylabel('%s − %s (mm / h)' % (datalab['imerg'], datalab['pcmk']))
    for ii, ids in enumerate(diff_satids):
        plt.text(ii, plt.axis()[3] + 0.4, '%d' % len(ids), ha = 'center')
    plt.grid(axis = 'y')
    plt.savefig('fig_sat_violin.pdf')
    plt.close()


if 'sat_case' in option:

    from glob import glob
    import h5py
    import gzip
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Rectangle
    from matplotlib.colors import BoundaryNorm

    xdis, ydis = 9, 10

    def plot_map(P, vmax, axes = 'imerg', plot_pcolor = True, 
                 cmap = plt.cm.viridis_r):

        if   axes == 'imerg':
            x, y = xedges1, yedges1
        elif axes == 'mrms':
            x, y = xedges2, yedges2

        if plot_pcolor:
            plt.pcolormesh(x, y, np.ma.masked_less(P, 0.1).T,
                           norm = BoundaryNorm(bounds, cmap.N),
                           cmap = cmap, rasterized = True)

        m = Basemap(projection='cyl', llcrnrlat = latb, urcrnrlat = latt,
            llcrnrlon = lonl, urcrnrlon = lonr, fix_aspect = True,
            resolution = 'i')
        _ = m.drawcoastlines(color = '0.5', linewidth = 0.5)

        plt.gca().add_patch(Rectangle((-75.6, 38), 0.1, 0.1, facecolor = 'none', 
                            linewidth = 1, edgecolor = '0.5'))

        return None

    inpath = '%s%4d/%02d/%02d/' % (imergpath, 2014, 4, 1)
    files = sorted(glob('%s3B-HHR*' % inpath))
    nt = len(files)

    with h5py.File(files[0]) as f:
        lats = f['Grid/lat'][:]
        lons = f['Grid/lon'][:]
        fillvalue = f['Grid/precipitationCal'].attrs['_FillValue']

    lat = np.where(lats ==  38.05)[0][0]
    lon = np.where(lons == -75.55)[0][0]

    # lat/lon edges of the zoomed out area
    latb = lats[lat - ydis] - 0.05
    latt = lats[lat + ydis] + 0.05
    lonl = lons[lon - xdis] - 0.05
    lonr = lons[lon + xdis] + 0.05

    # grid box edges for IMERG and MRMS grids
    xedges1 = np.linspace(lonl, lonr, (xdis + 1) * 2)
    yedges1 = np.linspace(latb, latt, (ydis + 1) * 2)
    xedges2 = np.linspace(lonl, lonr, xdis * 20 + 10 + 1)
    yedges2 = np.linspace(latb, latt, ydis * 20 + 10 + 1)

    # read the data

    imergfile1 = '3B-HHR.MS.MRG.3IMERG.20140716-S100000-E102959.0600.V03D.HDF5'
    imergfile2 = '3B-HHR.MS.MRG.3IMERG.20140716-S103000-E105959.0630.V03D.HDF5'

    inpath = '%s2014/07/16/' % imergpath

    with h5py.File(inpath + imergfile1, 'r') as f:
        P0 = f['Grid/precipitationCal'][lon - xdis : lon + xdis + 1, 
                                        lat - ydis : lat + ydis + 1]
    with h5py.File(inpath + imergfile2, 'r') as f:
        P1 = f['Grid/precipitationCal'][lon - xdis : lon + xdis + 1, 
                                        lat - ydis : lat + ydis + 1]
        Phq1 = f['Grid/HQprecipitation'][lon - xdis : lon + xdis + 1, 
                                         lat - ydis : lat + ydis + 1]

    Pimerg_large = 0.5 * (np.ma.masked_values(P0, fillvalue) + 
                          np.ma.masked_values(P1, fillvalue))
    Pimerg_hq = np.ma.masked_values(Phq1, fillvalue)

    mrmsfile1 = '1HGCF.20140716.110000.asc.gz'
    mrmsfile2 = '1HRQI.20140716.110000.asc.gz'

    inpath = '%s2014/07/' % mrmspath

    x1, x2 = 5440 - xdis * 10, 5450 + xdis * 10
    y1, y2 = 6 + 1689 - ydis * 10, 6 + 1699 + ydis * 10

    with gzip.open(inpath + mrmsfile1, 'rb') as f:
        P0 = np.array([[float(ii) for ii in jj.split()[x1 : x2]] 
                        for jj in f.readlines()[y1 : y2]])
    with gzip.open(inpath + mrmsfile2, 'rb') as f:
        rqi = np.array([[float(ii) for ii in jj.split()[x1 : x2]] 
                         for jj in f.readlines()[y1 : y2]])

    Pmrms = np.ma.masked_where(rqi < 100, np.ma.masked_values(P0, -999.0))[::-1]

    # reduce MRMS to IMERG grids (and transpose it to [lon, lat])
    Pm2i = np.ma.array([[np.ma.mean(Pmrms[ii * 10 : (ii + 1) * 10, 
                                          jj * 10: (jj + 1) * 10])
                         for jj in range(xdis * 2 + 1)] 
                        for ii in range(ydis * 2 + 1)]).T

    gproffile = '2A.F16.SSMIS.GPROF2014v1-4.20140716-S092934-E111129.055426.V03A.HDF5'

    with h5py.File('%s2014/07/16/%s' % (gprofpath, gproffile), 'r') as f:
        Pgprof = np.ma.masked_less(f['S1/surfacePrecipitation'][:], -100)
        gprof_lats = f['S1/Latitude'][:]
        gprof_lons = f['S1/Longitude'][:]

    # plot the maps

    bounds = np.array((0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2))

    plt.figure(figsize = (dcol, 0.75 * scol))

    plt.subplot(141)
    plot_map(Pm2i, bounds, 'imerg')
    plt.title('MRMS')
    plt.subplot(142)
    plot_map(Pimerg_large, bounds, 'imerg')
    plt.title('IMERG')
    plt.subplot(143)
    plot_map(Pimerg_hq, bounds, 'imerg')
    plt.title('HQprecipitation')
    plt.subplot(144)
    mp = plt.pcolormesh(gprof_lons, gprof_lats, np.ma.masked_less(Pgprof, 0.1), 
                        norm = BoundaryNorm(bounds, plt.cm.viridis_r.N),
                        cmap = plt.cm.viridis_r, rasterized = True)
    plot_map(Pgprof, bounds, plot_pcolor = False)
    plt.title('GPROF (SSMIS F16)')
    cb = plt.colorbar(mappable = mp, cax = plt.axes([0.125, 0.1, 0.775, 0.03]),
                      orientation = 'horizontal', ticks = bounds, 
                      extend = 'max')
    cb.set_label('mm / h')
    plt.savefig('fig_sat_case.pdf')
    plt.close()


#--- SUPPLEMENTARY ---#


if 'sup_case_morphing' in option:

    import h5py
    import os
    import gzip
    from datetime import timedelta
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import BoundaryNorm
    from matplotlib.patches import Rectangle

    year, month, day, timestep0 = 2014, 7, 16, 12
    nstep = 20
    xdis, ydis = 9, 10         # no. of IMERG grids to enlarge
    delay = 25                 # animation frames delay (units of 0.01 s)

    cmap1 = plt.cm.YlGnBu      # colormap for rain rates
    cmap2 = plt.cm.Accent      # colormap for HQprecipSource

    hour0 = timestep0 // 2
    minute0 = (timestep0 % 2) * 30

    Pcal = np.ma.masked_all([nstep, xdis * 2 + 1, ydis * 2 + 1])
    Psrc = np.ma.masked_all([nstep, xdis * 2 + 1, ydis * 2 + 1])
    Pmrms = np.ma.masked_all([nstep, xdis * 2 + 1, ydis * 2 + 1])
    times = []    # for collecting the times (to use in plots)

    for tt in range(nstep):

        t = (datetime(year, month, day, hour0, minute0) + 
             timedelta(minutes = 30 * tt))
        hour, minute = t.hour, t.minute

        times.append('%4d %02d %02d %02d%02d–%02d%02d' % (year, month, day, 
                     hour, minute, hour, minute + 29))

        # time labels for IMERG
        t = datetime(year, month, day, hour, minute)
        t0 = t.strftime('%H%M%S')
        t1 = (t + timedelta(seconds = 1799)).strftime('%H%M%S')
        t2 = (t - datetime(year, month, day)).total_seconds() / 60

        # time labels for MRMS
        t = datetime(year, month, day, hour)
        t3 = (t + timedelta(hours = 1)).strftime('%Y%m%d.%H%M%S')
        t4 = t.strftime('%H%M')
        t5 = (t + timedelta(hours = 1)).strftime('%H%M')

        inpath1 = '%s%4d/%02d/%02d/' % (imergpath, year, month, day)
        imergfile = '3B-HHR.MS.MRG.3IMERG.%4d%02d%02d-S%s-E%s.%04d.V03D.HDF5' \
                    % (year, month, day, t0, t1, t2)
        inpath2 = '%s%4d/%02d/' % (mrmspath, year, month)
        mrmsfile1 = '1HGCF.%s.asc.gz' % (t3,)
        mrmsfile2 = '1HRQI.%s.asc.gz' % (t3,)

        # read the IMERG data
        with h5py.File('%s%s' % (inpath1, imergfile)) as f:
            lats = f['Grid/lat'][:]
            lons = f['Grid/lon'][:]
            fillvalue = f['Grid/precipitationCal'].attrs['_FillValue']

            lat = np.where(lats ==  38.05)[0][0]
            lon = np.where(lons == -75.55)[0][0]

            Pcal[tt] = f['Grid/precipitationCal'][lon - xdis : lon + xdis + 1, 
                                              lat - ydis : lat + ydis + 1]
            Psrc[tt] = f['Grid/HQprecipSource'][lon - xdis : lon + xdis + 1, 
                                            lat - ydis : lat + ydis + 1]
            Pcal[tt] = np.ma.masked_values(Pcal[tt], fillvalue)
            Psrc[tt] = np.ma.masked_values(Psrc[tt], fillvalue)


        # read the MRMS data
        lats_mrms = np.arange(54.995, 20, -0.01)
        lons_mrms = np.arange(-129.995, -60, 0.01)
        lat0 = np.where(np.isclose(lats_mrms, 38.095))[0][0]
        lat1 = np.where(np.isclose(lats_mrms, 37.995))[0][0]
        lon0 = np.where(np.isclose(lons_mrms, -75.595))[0][0]
        lon1 = np.where(np.isclose(lons_mrms, -75.495))[0][0]

        with gzip.open('%s%s' % (inpath2, mrmsfile1), 'rb') as f:
            pre = np.array([[float(ii) for ii in jj.split()[lon0 - xdis * 10 : 
                             lon1 + xdis * 10]] for jj in f.readlines()[6 + 
                             lat0 - ydis * 10 : 6 + lat1 + ydis * 10]])
        with gzip.open('%s%s' % (inpath2, mrmsfile2), 'rb') as f:
            rqi = np.array([[float(ii) for ii in jj.split()[lon0 - xdis * 10 : 
                             lon1 + xdis * 10]] for jj in f.readlines()[6 + 
                             lat0 - ydis * 10 : 6 + lat1 + ydis * 10]])
        Pmrms_raw = np.ma.masked_values(np.ma.masked_where(rqi < 100, pre), 
                                        -999.0)[::-1]

        # reduce MRMS to IMERG grids (and transpose it to [lon, lat])
        for ii in range(xdis * 2 + 1):
            for jj in range(ydis * 2 + 1):
                Pslice = Pmrms_raw[jj * 10 : (jj + 1) * 10, 
                                   ii * 10 : (ii + 1) * 10]
                if np.ma.count(Pslice):
                    Pmrms[tt, ii, jj] = np.ma.mean(Pslice)

    def plot_map(x, y, P, bounds = False, cmap = plt.cm.YlGnBu):

        m.drawcoastlines(color = '0.5', zorder = 2)
        m.drawstates(color = '0.5', zorder = 2)

        if len(bounds) > 1:
            mp = plt.pcolormesh(x, y, P.T, cmap = cmap, zorder = 1,
                                norm = BoundaryNorm(bounds, cmap.N))
        else:
            mp = plt.pcolormesh(x, y, P.T, cmap = cmap, zorder = 1,
                                vmin = 0, vmax = bounds[0])

        plt.gca().add_patch(Rectangle((-75.6, 38), 0.1, 0.1, facecolor = 'none', 
                            linewidth = 1, edgecolor = '0.5'))

        return mp

    # lat/lon edges of area
    latb = lats[lat - ydis] - 0.05
    latt = lats[lat + ydis] + 0.05
    lonl = lons[lon - xdis] - 0.05
    lonr = lons[lon + xdis] + 0.05

    # create a Basemap instance for repeated use later (saves lots of time)
    m = Basemap(projection='cyl', llcrnrlat = latb, urcrnrlat = latt,
        llcrnrlon = lonl, urcrnrlon = lonr, fix_aspect = True,
        resolution = 'i')

    # grid box edges for IMERG
    xedges = np.linspace(lonl, lonr, (xdis + 1) * 2)
    yedges = np.linspace(latb, latt, (ydis + 1) * 2)

    # bounds of rain rates
    bounds1 = (50,)
    bounds2 = np.arange(1, 16) - 0.5

    # labels
    Plab = ('MRMS', 'IMERG')
    satlab = ('TMI', 'TCI', 'AMSR', 'SSMI', 'SSMIS', 'AMSU', 'MHS',
              'M.-T.', 'GMI', 'GCI', 'ATMS', 'AIRS', 'TOVS', 'CrlS')

    # plot the individual images

    outfiles = []    # for collecting the temp file names

    for tt in range(nstep):

        plt.figure(figsize = (dcol, 0.75 * dcol))
        plt.figtext(0.5, 0.925, times[tt], fontsize = 12, 
                    ha = 'center', va = 'bottom')

        for pp, P in enumerate((Pmrms, Pcal)):
            plt.subplot(231 + pp)
            mp = plot_map(xedges, yedges, np.ma.masked_less(P[tt], 0.1), 
                          bounds1, cmap1)
            if pp == 0:
                plt.ylabel('rain rate')
            plt.title('%s' % Plab[pp])
        cb = plt.colorbar(cax = plt.axes([0.125, 0.52, 0.5, 0.015]),
                          mappable = mp, orientation = 'horizontal',
                          extend = 'max',
                          ticks = [0.1] + list(np.linspace(0, 50, 6)[1:]))
        cb.set_label('mm / h')

        plt.subplot(233)
        mp = plot_map(xedges, yedges, np.ma.masked_equal(Psrc[tt], 0), 
                      bounds2, cmap2)
        plt.title('HQprecipSource')
        cb = plt.colorbar(cax = plt.axes([0.925, 0.12, 0.015, 0.76]),
                          mappable = mp, ticks = bounds2[:-1] + 0.5)
        cb.set_ticklabels(satlab)

        for pp, P in enumerate((Pmrms, Pcal)):
            plt.subplot(234 + pp)
            mp = plot_map(xedges, yedges, 
                          np.ma.masked_less(np.cumsum(P, 0)[tt] / 2, 0.1),
                          bounds1, cmap1)
            if pp == 0:
                plt.ylabel('rain accum.')
            plt.title(Plab[pp])
        cb = plt.colorbar(cax = plt.axes([0.125, 0.08, 0.5, 0.015]),
                          mappable = mp, orientation = 'horizontal', 
                          extend = 'max',
                          ticks = [0.1] + list(np.linspace(0, 50, 6)[1:]))
        cb.set_label('mm')

        outfile = 'tmp_%02d%02d%02d_%02d.png' % (year % 100, month, day, tt)
        outfiles.append(outfile)

        plt.savefig(outfile, bbox_inches = 'tight')
        plt.close()

    # animate the images

    t = datetime(year, month, day, hour, minute)
    ts = t.strftime('%H%M')
    te = (t + timedelta(minutes = 30 * nstep)).strftime('%H%M')

    giffile = 'sup_case_morphing.gif'
    command1 = 'convert -loop 0 -delay %d %s %s' % (delay, 
                ' '.join(outfiles), giffile)
    command2 = 'convert %s \( +clone -set delay %d \) +swap +delete %s' % (
                giffile, delay * 4, giffile)
    os.system(command1)
    os.system(command2)

    for outfile in outfiles:
        os.remove(outfile)


if 'sup_case_gsmap' in option:

    import h5py
    import gzip
    import struct
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Rectangle
    from matplotlib.colors import BoundaryNorm

    xdis, ydis = 9, 10

    gsmappath = '/media/jackson/Vault/GSMaP/2014/07/16/'

    imergfile1 = '3B-HHR.MS.MRG.3IMERG.20140716-S100000-E102959.0600.V03D.HDF5'
    imergfile2 = '3B-HHR.MS.MRG.3IMERG.20140716-S103000-E105959.0630.V03D.HDF5'
    gsmapfile = 'gsmap_gauge.20140716.1100.v6.0000.0.dat.gz'
    mrmsfile1 = '1HGCF.20140716.110000.asc.gz'
    mrmsfile2 = '1HRQI.20140716.110000.asc.gz'
    gproffile = '2A.F16.SSMIS.GPROF2014v1-4.20140716-S092934-E111129.055426.V03A.HDF5'

    # read IMERG
    with h5py.File('%s2014/07/16/%s' % (imergpath, imergfile1), 'r') as f:
        lats = f['Grid/lat'][:]
        lons = f['Grid/lon'][:]
        fillvalue = f['Grid/precipitationCal'].attrs['_FillValue']
        
    lat = np.where(lats ==  38.05)[0][0]
    lon = np.where(lons == -75.55)[0][0]

    with h5py.File('%s2014/07/16/%s' % (imergpath, imergfile1), 'r') as f:
        P0 = f['Grid/precipitationCal'][lon - xdis : lon + xdis + 1, 
                                        lat - ydis : lat + ydis + 1]

    with h5py.File('%s2014/07/16/%s' % (imergpath, imergfile2), 'r') as f:
        P1 = f['Grid/precipitationCal'][lon - xdis : lon + xdis + 1, 
                                        lat - ydis : lat + ydis + 1]

    Pimerg = np.ma.masked_values(0.5 * (P0 + P1), fillvalue)

    # read GSMaP

    lats_gsmap = np.arange(59.95, -60, -0.1)
    lons_gsmap = np.arange(0.05, 360, 0.1)

    lat_gsmap = np.where(np.isclose(lats_gsmap,  38.05))[0][0]
    lon_gsmap = np.where(np.isclose(lons_gsmap, 284.45))[0][0]

    with gzip.open('%s%s' % (gsmappath, gsmapfile), 'rb') as f:
        raw = f.read()
        Pgsmap_raw = struct.unpack('<' + 'f' * (len(raw) // 4), raw)

    Pgsmap_all = np.ma.masked_less(Pgsmap_raw, 0).reshape(1200, 3600)
    Pgsmap = Pgsmap_all[lat_gsmap - ydis : lat_gsmap + ydis + 1, 
                        lon_gsmap - xdis : lon_gsmap + xdis + 1][::-1].T

    # read MRMS
    x1, x2 = 5440 - xdis * 10, 5450 + xdis * 10
    y1, y2 = 6 + 1689 - ydis * 10, 6 + 1699 + ydis * 10

    with gzip.open(inpath + mrmsfile1, 'rb') as f:
        P0 = np.array([[float(ii) for ii in jj.split()[x1 : x2]] 
                        for jj in f.readlines()[y1 : y2]])
    with gzip.open(inpath + mrmsfile2, 'rb') as f:
        rqi = np.array([[float(ii) for ii in jj.split()[x1 : x2]] 
                         for jj in f.readlines()[y1 : y2]])
    Pmrms = np.ma.masked_where(rqi < 100, np.ma.masked_values(P0[::-1], -999.0))

    # reduce MRMS to IMERG grids
    Pm2i = np.ma.array([[np.ma.mean(Pmrms[ii * 10 : (ii + 1) * 10, 
                                          jj * 10: (jj + 1) * 10])
                         for jj in range(xdis * 2 + 1)] 
                        for ii in range(ydis * 2 + 1)]).T

    # read GPROF.
    with h5py.File('%s2014/07/16/%s' % (gprofpath, gproffile), 'r') as f:
        Pgprof = np.ma.masked_less(f['S1/surfacePrecipitation'][:], -100)
        gprof_lats = f['S1/Latitude'][:]
        gprof_lons = f['S1/Longitude'][:]

    # lat/lon edges of the zoomed out area
    latb = lats[lat - ydis] - 0.05
    latt = lats[lat + ydis] + 0.05
    lonl = lons[lon - xdis] - 0.05
    lonr = lons[lon + xdis] + 0.05

    # grid box edges for IMERG and MRMS grids
    xedges1 = np.linspace(lonl, lonr, (xdis + 1) * 2)
    yedges1 = np.linspace(latb, latt, (ydis + 1) * 2)
    xedges2 = np.linspace(lonl, lonr, xdis * 20 + 10 + 1)
    yedges2 = np.linspace(latb, latt, ydis * 20 + 10 + 1)

    # make the plot

    def plot_map(P, bounds, axes = 'imerg', plot_pcolor = True, 
                 cmap = plt.cm.viridis_r):

        if   axes == 'imerg':
            x, y = xedges1, yedges1
        elif axes == 'mrms':
            x, y = xedges2, yedges2

        if plot_pcolor:
            plt.pcolormesh(x, y, np.ma.masked_less(P, 0.1).T,
                           norm = BoundaryNorm(bounds, cmap.N),
                           cmap = cmap, rasterized = True)

        m = Basemap(projection='cyl', llcrnrlat = latb, urcrnrlat = latt,
            llcrnrlon = lonl, urcrnrlon = lonr, fix_aspect = True,
            resolution = 'i')
        _ = m.drawcoastlines(color = '0.5', linewidth = 0.5)

        plt.gca().add_patch(Rectangle((-75.6, 38), 0.1, 0.1, facecolor = 'none', 
                            linewidth = 1, edgecolor = '0.5'))

        return None

    # plot the maps

    bounds = np.array((0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2))

    plt.figure(figsize = (dcol, 0.75 * scol))
    #plt.suptitle('2014 07 16 1000−1100', fontsize = 12)

    plt.subplot(141)
    plot_map(Pm2i, bounds, 'imerg')
    plt.title('MRMS')
    plt.subplot(142)
    mp = plt.pcolormesh(gprof_lons, gprof_lats, np.ma.masked_less(Pgprof, 0.1), 
                        norm = BoundaryNorm(bounds, plt.cm.YlGnBu.N),
                        cmap = plt.cm.viridis_r, rasterized = True)
    plot_map(Pgprof, bounds, plot_pcolor = False)
    plt.title('GPROF (SSMIS F16)')
    plt.subplot(143)
    plot_map(Pimerg, bounds, 'imerg')
    plt.title('IMERG')
    plt.subplot(144)
    plot_map(Pgsmap, bounds, 'imerg')
    plt.title('GSMaP')

    cb = plt.colorbar(mappable = mp, cax = plt.axes([0.125, 0.1, 0.775, 0.03]),
                      orientation = 'horizontal', ticks = bounds, 
                      extend = 'max')
    cb.set_label('mm / h')
    plt.savefig('sup_case_gsmap.pdf')
    plt.close()


#--- ADDITIONAL ---#


if 'add_all_scatter' in option:

    P = read_data_uncal(year0, month0, year1, month1)

    pairs = (('mrms2', 'mrms1'), ('pcmk', 'mrms2'), 
             ('pcmk', 'imerg'))

    t = np.logspace(-1, 2, 101)

    plt.figure(figsize = (dcol, 0.5 * scol))
    plt.subplots_adjust(wspace = 0.4)
    for n, (x, y) in enumerate(pairs):
        plt.subplot(131 + n)
        scatter_log(P['%s_1' % x], P['%s_1' % y], size = 3,
                    threshold = minrain)
        alpha, beta, sigma = fit_mem(P['%s_1' % x], P['%s_1' % y], minrain)
        plt.plot(t, np.exp(alpha) * t ** beta, 'k')
        plt.xlabel('%s (mm / h)' % datalab[x])
        plt.ylabel('%s (mm / h)' % datalab[y])
        plt.text(0.03, 0.97, subplab[n], ha = 'left', va = 'top', 
            transform = plt.gca().transAxes)
    plt.savefig('add_all_scatter.pdf')
    plt.close()


if 'add_sat_contab' in option:

    sat_imerg, sat_pcmk, _ = read_platform_data_uncal(year0, month0, year1, month1)

    hit, miss, false, correj = [], [], [], []

    for ss in range(len(satlab)):
        hit.append(np.sum(compare(np.array(sat_pcmk[ss]), 'Y', minrain) * 
                          compare(np.array(sat_imerg[ss]), 'Y', minrain)))
        miss.append(np.sum(compare(np.array(sat_pcmk[ss]), 'Y', minrain) * 
                           compare(np.array(sat_imerg[ss]), 'N', minrain)))
        false.append(np.sum(compare(np.array(sat_pcmk[ss]), 'N', minrain) * 
                            compare(np.array(sat_imerg[ss]), 'Y', minrain)))
        correj.append(np.sum(compare(np.array(sat_pcmk[ss]), 'N', minrain) * 
                             compare(np.array(sat_imerg[ss]), 'N', minrain)))

    def hss(H, M, F, C):
        N = H + M + F + C
        exp = 1 / N * ((H + M) * (H + F) + (C + M) * (C + F))
        return ((H + C - exp) / (N - exp))

    print('HSS for IR only: %5.3f' % hss(hit[0], miss[0], false[0], correj[0]))

    counts = np.vstack([hit, miss, false, correj])
    rawfrac = (counts / np.sum(counts, 0)) * 100

    barmax = 30
    frac = rawfrac.copy()
    frac[3] = frac[3] - (100 - barmax)

    contablab = ('hit', 'miss', 'false alarm', 'correct negative')

    y = np.arange(len(satlab))

    plt.figure(figsize = (0.75 * dcol, 0.5 * scol))
    for ii in range(4):
        plt.barh(y, frac[ii], color = cols[ii], left = np.sum(frac[0 : ii], 0),
                align = 'center', label = contablab[ii], edgecolor = 'none')
    plt.yticks(y)
    plt.setp(plt.gca(), 'yticklabels', satlab)
    plt.xlabel('(%)')
    plt.axis([0, barmax + 2, -0.4, len(satlab) - 1 + 0.4])
    plt.legend(loc = 1)
    for cc, count in enumerate(counts.T):
        polygon = plt.Polygon(([barmax + 2, cc], [barmax - 0.1, cc - 0.4], 
                               [barmax - 0.1, cc + 0.4]),
                              edgecolor = 'none', facecolor = cols[3])
        plt.gca().add_patch(polygon)
        plt.text(plt.axis()[1] + 3.0, cc, '{0:5d}'.format(np.sum(count)),
                 ha = 'right', va = 'center')
        plt.text(plt.axis()[1] + 6.7, cc, '({0:4.1f}%)'.format(np.sum(count) / 
                 np.sum(counts) * 100), ha = 'right', va = 'center')
        for ii in range(4):
            plt.text(np.sum(frac[:ii, cc]) + frac[ii, cc] / 2, cc,
                     '%4.1f' % rawfrac[ii, cc], color = 'w', fontsize = 7,
                     ha = 'center', va = 'center')

    plt.savefig('add_sat_contab.pdf')
    plt.close()

    #np.savetxt('fig_sat_contab.txt', rawfrac.T, fmt = '%4.1f')


if 'add_sat_scatter' in option:

    sat_imerg, sat_pcmk, _ = read_platform_data_uncal(year0, month0, year1, month1)

    t = np.logspace(-1, 2, 101)

    plt.figure(figsize = (dcol, dcol))
    plt.subplots_adjust(hspace = 0.25, wspace = 0.25)

    for ss in range(len(satlab)):

        plt.subplot(331 + ss)
        scatter_log(np.array(sat_pcmk[ss]), np.array(sat_imerg[ss]), 
                    size = 5, threshold = minrain)

        if ss:    # skip IR because of only one point
            alpha, beta, sigma = fit_mem(np.array(sat_pcmk[ss]), 
                                         np.array(sat_imerg[ss]),
                                         minrain)
            plt.plot(t, np.exp(alpha) * t ** beta, 'k')

        plt.text(0.03, 0.97, satlab[ss], ha = 'left', va = 'top', 
            transform = plt.gca().transAxes)

        if ss > 4: plt.xlabel('%s (mm / h)' % datalab['pcmk'])
        if not ss % 3: plt.ylabel('%s (mm / h)' % datalab['imerg'])

        if   ss < 3:
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position("top")
        elif ss < 6:
            plt.gca().set_xticklabels([])
            plt.xlabel('')

        if   not (ss - 2) % 3:
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position("right")
        elif not (ss - 1) % 3:
            plt.gca().set_yticklabels([])
            plt.ylabel('')

    plt.savefig('add_sat_scatter.pdf')
    plt.close()
