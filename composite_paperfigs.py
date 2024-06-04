##########################################################################
#
#  COMPOSITE PAPERFIGS
#
#  Drawing on the functions developed in combined_analysis, 
#  but building the multipanel composite figs I use in the paper.
#
##########################################################################

import utilities

import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from matplotlib.ticker import LogFormatter 

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

import yaml
import glob
import calendar
from tqdm import tqdm
from datetime import datetime

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings('ignore',category=ShapelyDeprecationWarning)
warnings.filterwarnings('ignore',message='Input array is not C_CONTIGUOUS')
warnings.filterwarnings('ignore',message='invalid value encountered in divide')
warnings.filterwarnings('ignore',message='The handle <matplotlib.lines.Line2D object')
warnings.filterwarnings('ignore',message='The label \'_child')


##########################################################################
#  Define Regions
##########################################################################

# new defs based on SREX regions
regionlims = {'Global':        [0,360,-90,90],  # not using SREX
              'N. Hemis':      [0,360,0,90],    # not using SREX
              'Global (60S-60N)': [0,360,-60,60], # not using SREX
              'N. Hemis (0-60N)': [0,360,0,60],   # not using SREX
              'East Asia':     [100,145,20,50], # SRREX EAS
              'South Asia':    [60,100,5,30],   # SREX SAS
              'Europe':        [0,35,35,60],    # not using SREX
              'USA':           [235,290,25,50], # not using SREX
              'C. Africa':     [0,40,-11,15],   # EAF and half of WAF (to avoid crossing lon=0)
              'S. America':    [280,325,-40,0]} # not using SREX


##########################################################################
#  AAOD / ERFari / AOD composite: maps, ts, spread
##########################################################################

def composite_maps_ts(var,table,units,testnonzero=False):

    print('composite map/ts/spread plot for '+var)

    # -- set up figure ----------------------------------------------------

    # set up figure
    f = plt.figure(figsize=(15,12))
    gs = f.add_gridspec(3,2)
    ax = [[f.add_subplot(gs[i,0],projection=ccrs.Robinson()),
           f.add_subplot(gs[i,1])] for i in range(3)]
    ax = np.array(ax)
    plt.subplots_adjust(left=0.02,right=0.96,bottom=0.04,top=0.96,hspace=0.35)

    # set colormaps for map panels
    c0 = {'abs550aer':'viridis','od550aer':'cividis','ERFari':'RdBu_r'}[var]
    cmap_val,cmap_dif = c0,'bwr'

    # set color ranges for map panels
    if var=='ERFari':
        lmin_val,lmax_val,dlev_val = -3, 3, 0.50
        lmin_dif,lmax_dif,dlev_dif = -1, 1, 0.25
    elif var=='abs550aer':
        lmin_val,lmax_val,dlev_val =  0.00, 0.03, 0.0030
        lmin_dif,lmax_dif,dlev_dif = -0.01, 0.01, 0.0025
    elif var=='od550aer':
        lmin_val,lmax_val,dlev_val =  0.00, 0.90, 0.10
        lmin_dif,lmax_dif,dlev_dif = -0.05, 0.05, 0.01
    clevs_val = np.arange(lmin_val,lmax_val+dlev_val,dlev_val)
    clevs_dif = np.arange(lmin_dif,lmax_dif+dlev_dif,dlev_dif)
    norm_val = colors.BoundaryNorm(np.arange(lmin_val-dlev_val/2,lmax_val+dlev_val*1.5,dlev_val),256)
    norm_dif = colors.BoundaryNorm(np.arange(lmin_dif-dlev_dif/2,lmax_dif+dlev_dif*1.5,dlev_dif),256)

    # set colours for timeseries 
    clist_ts = [plt.cm.plasma(i/3) for i in range(3)]

    # variable and unit strings for labels
    vstr = {'ERFari':'BC ERFari','abs550aer':'AAOD','od550aer':'AOD'}[var]
    ustr = ' [%s]'%units if type(units)==str else ''


    # -- read, process, plot sim data; store lo and hi for panel f --------------------

    ddict = {}
    for i,(ri,label) in enumerate(zip(['lo','mid','hi'],['low','medium','high'])):
                                      
        # read in data: 9 x 2015-2019
        if 'ERF' in var:
            ds = utilities.calc_forcing(var,'bc-ri%s-ctrl'%ri,'bc-ri%s-1850bc'%ri,n_ens=9)
            ds = ds.rename({'rf':var})
        else: 
            ds = utilities.read_sim_ens('bc-ri%s-ctrl'%ri,table,var,n_ens=9)
        ds = ds.sel(time=slice('2015','2019'))
        if i==0: ref = ds.mean('time')

        # save data to calculate spreads for panel f
        if i in [0,2]: ddict[ri] = ds.mean('time')

        # do t-tests as necessary (time mean, calc over run)
        if (i==0):
            if testnonzero:
                t,p_non0 = stats.ttest_1samp(a=ds[var].mean('time').values,popmean=0,axis=0,
                                            alternative='two-sided')
                msk_sig = np.ma.masked_where(p_non0>0.05,p_non0)
            else: 
                msk_sig = None
        elif i>0:
            t,p_rilo = stats.ttest_ind(a=ds[var].mean('time').values,b=ref[var].values,axis=0,
                                       equal_var=True,alternative='two-sided')
            msk_sig = np.ma.masked_where(p_rilo>0.05,p_rilo)

        # ax[i,0]: map (vals if i=0, vals-lo if i>0)
        mp = ds.mean('time').median('run')
        gm = mp[var].weighted(np.cos(np.deg2rad(mp.lat))).mean(('lat','lon')).values
        gms = '%.2f W/m2'%gm if var=='ERFari' else '%.2e'%gm
        prefix = 'low BCRI ensemble median' if i==0 else '%s minus low BCRI'%label
        ax[i,0].set_title('%s \n global mean = %s'%(prefix,gms),fontweight='bold')
        if i==0: 
            cmap,clevs,norm = cmap_val, clevs_val, norm_val
        else: 
            mp = mp - ref.median('run')
            cmap,clevs,norm = cmap_dif, clevs_dif, norm_dif
        im = ax[i,0].pcolor(mp.lon,mp.lat,mp[var],transform=ccrs.PlateCarree(),
                            cmap=cmap,norm=norm,shading='auto',rasterized=True)
        if msk_sig is not None: 
            ax[i,0].pcolor(mp.lon,mp.lat,msk_sig,transform=ccrs.PlateCarree(),
                           hatch='..',alpha=0,rasterized=True)
        ax[i,0].coastlines()

        # ax[i,0]: colorbar
        cb = f.colorbar(im,ax=ax[i,0],label=vstr+ustr,location='bottom',
                        drawedges=False,ticks=clevs[::2],
                        aspect=30,fraction=0.05,pad=0.02,extend='both')
        cb.ax.set_xticks([],minor=True)

        # ax[1,1]: 2015-2019 ts (monthly mean)
        ax[1,1].set_title('global mean %s (2015-2019 monthly mean)'%vstr,fontweight='bold')
        ts = ds[var].weighted(np.cos(np.deg2rad(ds.lat))).mean(('lat','lon'))
        time = [np.datetime64(t) for t in ts.time.values]
        ax[1,1].plot(time,ts.median('run'),c=clist_ts[i],lw=1.5,label=label)
        ax[1,1].fill_between(time,ts.quantile(0.05,dim='run'),
                             ts.quantile(0.95,dim='run'),
                             color=clist_ts[i],alpha=0.7)

        # ax[0,1]: 1950-2019 ts (ann means): prep to combine with long run
        ts_shortens = ts.groupby(ts.time.dt.year).mean('time')

        # read in data: 1 x 1950-2019
        if 'ERF' in var:
            dsl = utilities.calc_forcing(var,'bc-ri%s-ctrl-long'%ri,
                                         'bc-ri%s-1850bc-long'%ri,n_ens=1) \
                           .rename({'rf':var})
        else: 
            dsl = utilities.read_sim_ens('bc-ri%s-ctrl-long'%ri,table,var,n_ens=1)
        dsl = dsl.groupby(dsl.time.dt.year).mean('time').sel(year=slice(1950,2019))
        dsl = dsl[var].weighted(np.cos(np.deg2rad(ds.lat))).mean(('lat','lon'))
        year_long = dsl.year.values

        # ax[0,1], 1950-2015: long run only
        ax[0,1].set_title('global mean %s (1950-2019 annual mean)'%vstr,fontweight='bold')
        ax[0,1].plot(year_long[:-4],dsl.values[:-4],c=clist_ts[i],lw=1.5,label=label)

        # ax[0,1], 1950-2015: add 11-year rolling median (ERFari only)
        if var=='ERFari':
            dsmooth = dsl.rolling(year=11,center=True).mean().dropna('year')
            tsmooth = dsmooth.year.values
            ax[0,1].plot(tsmooth,dsmooth.values,c=clist_ts[i],lw=2.5)


        # ax[0,1], 2015-2019: combine long run with shortens
        ts_short = np.concatenate([ts_shortens.values,np.reshape(dsl.values[-5:],(1,5))])
        ax[0,1].plot(year_long[-5:],np.median(ts_short,axis=0),c=clist_ts[i],lw=1.5)
        ax[0,1].fill_between(year_long[-5:],np.percentile(ts_short,5,axis=0),
                             np.percentile(ts_short,95,axis=0),color=clist_ts[i],alpha=0.7)

        # close ds's
        for dsi in [ds,dsl,ts,ts_shortens]: dsi.close()


    # -- compare spreads from different sources: AAOD & AOD (by region) ------------------
    
    if var in ['abs550aer','od550aer']:

        regions = list(regionlims.keys())[2:]

        # alternate aerosol emissions
        dsinv = utilities.read_sim_file('bc-rilo-ctrl-oldinv',table,var) \
                         .sel(time=slice('2015','2019')).mean('time')   
        ddict['oldinv'] = dsinv

        # observations
        if var=='abs550aer':
            dsobs = utilities.read_obs_data('abs550aer',2007,2011).mean('time') 
            ddict['satnames'] = ['MISR','POLDER-GRASP']
        elif var=='od550aer':
            dsobs = utilities.read_obs_data('od550aer',2015,2019).mean('time') 
            ddict['satnames'] = ['MISR','MODIS Aqua','MODIS Terra','CALIOP']
        ddict['satobs'] = [dsobs.sel(sat=sat) for sat in ddict['satnames']]

        # colors and labels
        labels_bar = ['low to high BCRI','emission inventory','satellite choice','AeroComIII']
        clist_bar = [plt.cm.plasma(0.5),plt.cm.Wistia(0.5),plt.cm.gist_earth(0.3),'darkslateblue']
        labels_reg = []
        for reg in regions: 
            split = reg.split(' ')
            if   len(split)==1: labels_reg.extend([reg])
            elif len(split)==2: 
                if len(split[0])<3: labels_reg.extend([reg]) 
                else: labels_reg.extend([split[0]+'\n'+split[1]])
            elif len(split)==3: labels_reg.extend([split[0]+' '+split[1]+'\n'+split[2]])
        nk,nr = len(labels_bar),len(labels_reg)

        # iterate through regions 
        for i,region in enumerate(regions):

            # calculate region-means and resulting difs
            lims = regionlims[region]
            lo,hi,old = [dsi.sel(lon=slice(lims[0],lims[1]),lat=slice(lims[2],lims[3]))
                         for dsi in [ddict['lo'],ddict['hi'],ddict['oldinv']]]
            lo,hi,old = [dsi[var].weighted(np.cos(np.deg2rad(dsi.lat))).mean(('lat','lon'))
                         for dsi in [lo,hi,old]]
            obslist = [dsi.sel(lon=slice(lims[0],lims[1]),lat=slice(lims[2],lims[3]))
                       for dsi in ddict['satobs']]
            obslist = [dsi[var].weighted(np.cos(np.deg2rad(dsi.lat))).mean(('lat','lon'))
                       for dsi in obslist]
            diflist = [hi-lo, old-lo, max(obslist)-min(obslist)]
            # AeroComIII spread from Sand2021 fig2
            if (var=='abs550aer')*(region=='Global (60S-60N)'):
                diflist.extend([xr.DataArray(data=np.array(7.75e-3))])

            # plot bars
            for j,dif in enumerate(diflist):
                if 'run' in dif.dims: 
                    d05 = dif.quantile(0.05,dim='run').values
                    d50 = dif.quantile(0.50,dim='run').values
                    d95 = dif.quantile(0.95,dim='run').values
                    bar = ax[2,1].bar(i*(nk+1)+j,d50,fc=clist_bar[j],yerr=[[d50-d05],[d95-d50]],
                                    label=labels_bar[j] if i==0 else '')
                else: 
                    bar = ax[2,1].bar(i*(nk+1)+j,dif.values,fc=clist_bar[j],
                                      label=labels_bar[j] if i==0 else '')
                dif.close()

        # ax[2,1]: labels
        ax[2,1].set_title('regional %s variation from different sources'%vstr,fontweight='bold')
        midpts = (nk-1)/2. + np.array([(nk+1)*i for i in range(nr)])
        ax[2,1].set_xticks(midpts,labels=labels_reg)


    # -- compute spreads from different sources: ERFari (global only) ------------------

    elif var=='ERFari':

        bcri = ddict['hi'] - ddict['lo']
        bcri = bcri[var].weighted(np.cos(np.deg2rad(bcri.lat))).mean(('lat','lon'))
        bcri_05 = bcri.quantile(0.05,'run').values
        bcri_50 = bcri.quantile(0.50,'run').values
        bcri_95 = bcri.quantile(0.95,'run').values

        thornhill_range = 0.24
        amap_unc = 0.08

        clist_bar = ['steelblue','tan']
        labels_bar = ['Thornhill (2021) \nmultimodel range','AMAP (2021) \nstat. uncertainty']
        vals_bar = [thornhill_range,amap_unc]

        ax[2,1].set_title('%s sensitivity comparison'%vstr,fontweight='bold')
        ax[2,1].bar(0,bcri_50,yerr=[[bcri_95-bcri_50],[bcri_50-bcri_05]],
                    fc=plt.cm.plasma(0.5),label='low to high BCRI')
        for i in [0,1]: ax[2,1].bar(i+1,vals_bar[i],fc=clist_bar[i],label=labels_bar[i])
        ax[2,1].set_xticks([])

        bcri.close()

    
    # -- finish and save fig ----------------------------------------------------------

    # panel annotation
    for a,let in zip(ax.flat,'adbecf'):
        a.text(0.03,0.9,'(%s)'%let,transform=a.transAxes,fontweight='bold',fontsize=16)

    # legends
    if var=='ERFari':        
        for i in range(2):
            ax[i,1].legend(bbox_to_anchor=(0.1,1),loc='upper left',
                           bbox_transform=ax[i,1].transAxes)
        ax[2,1].legend(loc='upper right')
    else: 
        for i in range(3):
            ax[i,1].legend(bbox_to_anchor=(0.1,1),loc='upper left',
                           bbox_transform=ax[i,1].transAxes)
    # ylabels
    for i in range(2): ax[i,1].set_ylabel(vstr+ustr)
    ax[2,1].set_ylabel(vstr+' difference'+ustr)
    ax[2,1].axhline(0,c='k',ls='-')

    # save
    plt.savefig(pdf_plots,format='pdf')

    return

##########################################################################
#  Composite temperature 
##########################################################################

def composite_temp():

    print('composite temperature plot')

    # -- set up figure ----------------------------------------------------

    # set up figure
    f = plt.figure(figsize=(8,8))
    gs = f.add_gridspec(2,3,height_ratios=[1,1.25],width_ratios=[1,6,1])
    ax = [f.add_subplot(gs[0,1]),
          f.add_subplot(gs[1,:],projection=ccrs.Robinson())]
    ax = np.array(ax)
    plt.subplots_adjust(left=0.01,right=0.98,bottom=0.04,top=0.96,hspace=0.3)

    # color maps (only plotting hi-lo difference)
    cmap,lmin,lmax,dlev = 'bwr',-0.4,0.4,0.1
    clevs = np.arange(lmin,lmax+dlev,dlev)
    norm = colors.BoundaryNorm(np.arange(lmin-dlev/2,lmax+dlev*1.5,dlev),256)

    # read in data
    lo = utilities.read_sim_ens('bc-rilo-ctrl','Amon','ta',n_ens=9).sel(time=slice('2015','2019')).mean('time')
    hi = utilities.read_sim_ens('bc-rihi-ctrl','Amon','ta',n_ens=9).sel(time=slice('2015','2019')).mean('time')
    
    # panel 0: zonal mean vertical profile
    ax[0].set_title('high minus low BCRI',fontweight='bold')
    zml,zmh = lo.ta.mean('lon'), hi.ta.mean('lon')
    t,p = stats.ttest_ind(a=zmh.values,b=zml.values,axis=0,
                          equal_var=True,alternative='two-sided')
    ax[0].pcolor(zml.lat,zml.plev,zmh.median('run')-zml.median('run'),
                 cmap=cmap,norm=norm,shading='auto',rasterized=True)
    ax[0].pcolor(zml.lat,zml.plev,np.ma.masked_where(p>0.05,p),
                 hatch='..',alpha=0,rasterized=True)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('latitude',fontsize=10)
    ax[0].set_ylabel('hPa',fontsize=10)
    ax[0].set_yticks(np.arange(0,120000,20000),labels=np.arange(0,1200,200))
    for zm in [zml,zmh]: zm.close()

    # panel 1: map slice at 850hPa
    ax[1].set_title('difference at 850 hPa',fontweight='bold')
    mpl,mph = lo.ta.sel(plev=85000), hi.ta.sel(plev=85000)
    t,p = stats.ttest_ind(a=mph.values,b=mpl.values,axis=0,
                          equal_var=True,alternative='two-sided')
    im = ax[1].pcolor(mpl.lon,mpl.lat,mph.median('run')-mpl.median('run'),
                      transform=ccrs.PlateCarree(),
                      cmap=cmap,norm=norm,shading='auto',rasterized=True)
    ax[1].pcolor(mpl.lon,mpl.lat,np.ma.masked_where(p>0.05,p),
                 transform=ccrs.PlateCarree(),
                 hatch='..',alpha=0,rasterized=True)
    ax[1].coastlines()
    cb = f.colorbar(im,ax=ax[1],label='temperature difference [K]',location='bottom',
                    pad=0.1,shrink=0.8,drawedges=False,ticks=clevs[::2],extend='both')
    cb.ax.set_xticks([],minor=True)

    # annotate subplots
    ax[0].text(0.03,0.9,'(a)',transform=ax[0].transAxes,fontweight='bold',fontsize=16)
    ax[1].text(0.02,1.0,'(b)',transform=ax[1].transAxes,fontweight='bold',fontsize=16)
    
    plt.savefig(pdf_plots,format='pdf')

    return


##########################################################################
#  MAIN
##########################################################################

basepath = '/home/rud001/Analysis/Aerosols/black_carbon/'
pdf_plots = PdfPages(basepath+'plots/composite_paperfigs.pdf')

composite_maps_ts('abs550aer','AERmon', None,     testnonzero=False)
composite_maps_ts('ERFari',   '',      'W/m$^2$', testnonzero=True)
composite_temp()
composite_maps_ts('od550aer', 'AERmon', None,     testnonzero=False)

pdf_plots.close()