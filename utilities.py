########################################################################
#
#  Utilities for BC project
#
#  Reads files in/out, calculates forcings. 
#
#  This version contains psuedocode to maintain data storage privacy.
#
########################################################################

import numpy as np
import xarray as xr
import pandas as pd
import glob


########################################################################
#  Read in an ensemble's files
########################################################################

def read_sim_ens(ridbase,table,var,n_ens):

    if n_ens==1: 
        ds = read_sim_file(ridbase,table,var)

    # assign ensemble members "runids" of 01,02,03... 
    # instead of bc-rilo-ctrl-01, etc to faciliate 
    # subraction between experiments/RIvals
    elif n_ens>1:
        runnums = ['%02d'%n for n in np.arange(1,n_ens+1)]
        runids = ['%s-%s'%(ridbase,rnum) for rnum in runnums]
        dslist = [read_sim_file(runid,table,var) for runid in runids]
        ds = xr.concat(dslist,pd.Index(runnums,name='run')).chunk({'run':-1})
        
    return ds


########################################################################
#  Read in files given runid, table, and var
########################################################################

def read_sim_file(runid,table,var):

    dpath = 'function/of/runid/table/and/var/'
    fnames = glob.glob(dpath,recursive=True)

    # open: avoid mfdataset if poss, can make things slow
    if len(fnames)==1: ds = xr.open_dataset(fnames[0])

    # open: mfdataset if necessary (in a few cases, years are in ind files)
    else: ds = xr.open_mfdataset(fnames)

    # convert units
    if var in ['ta','tas']:         # originally deg K
        ds[var] = ds[var] - 273.15  # new: deg C

    return ds    


########################################################################
#  Coordinate the reading/preprocessing of reference obs data
#  Taken from preprocessed (incl regridded) files.
########################################################################

def read_obs_data(var,y0,yf):

    dpath = '/path/to/preprocessed/obs/' 

    ovar = {'od550aer':'aod','abs550aer':'aaod',
            'od550dust':'dod','ofst':'ae'}[var]
        
    fname = dpath+'%s_multisat_T63_2003-2020.nc'%ovar
    ds = xr.open_dataset(fname).sel(time=slice(str(y0),str(yf)))
    ds = ds.rename({ovar:var})

    return ds


########################################################################
#  Calculate flux: read in data for var(s); if there are 2, subtract.
########################################################################

def calc_flux(runid,tables,varlist):

    ds = read_sim_file(runid,tables[0],varlist[0]).rename({varlist[0]:'flux'})

    if len(varlist)==2:    
        ds2 = read_sim_file(runid,tables[1],varlist[1]).rename({varlist[1]:'flux'})
        ds = ds-ds2
        ds2.close()

    return ds


########################################################################
#  Calculate forcing components (single run or ensemble)
########################################################################

def calc_forcing(component,ctrlid,prtbid,n_ens):

    # calculate RF: subtract flux b/w control and perturbed runs
    # RF is positive for net downward flux
    # total_ERF is calculated from rtmt which gives downward flux
    # others are calculated from outgoing flux 
    # --> for these fields, do prtb-ctrl as equiv to (-ctrl) - (-prtb)

    rf_vars = {'ERF SW+LW': ['rtmt'],
               'ERFtot':  ['rsut'],
               'ERFari':  ['rsut','rsutaf'],
               'ERFaci':  ['rsutaf','rsutcsaf'],
               'ERFalb':  ['rsutcsaf']}
    var_tables = {var:'Amon' for var in ['rtmt','rsut','rsutcs','rlut']}
    for var in ['rsutaf','rsutcsaf']: var_tables[var] = 'AERmon'

    varlist = rf_vars[component]
    tables = [var_tables[var] for var in varlist]

    # if n_ens=1, do calculation for a single ctrl/prtb pair
    # if n_ens>1, this is an ensemble with runids of the form 
    #             ctrlid-NN and prtbid-NN.  

    if (n_ens==1)*('bes' not in ctrlid):
        ds_c = calc_flux(ctrlid,tables,varlist)
        ds_p = calc_flux(prtbid,tables,varlist)
    else: 
        ctrlids = ['%s-%02d'%(ctrlid,n) for n in np.arange(1,n_ens+1)]
        prtbids = ['%s-%02d'%(prtbid,n) for n in np.arange(1,n_ens+1)]
        outids = ['%02d'%n for n in np.arange(1,n_ens+1)] # to enable subtraction b/w expts
        dslist_c = [calc_flux(cid,tables,varlist) for cid in ctrlids]
        dslist_p = [calc_flux(pid,tables,varlist) for pid in prtbids]
        dslist_c = [ds.drop('time_bnds') for ds in dslist_c]
        dslist_p = [ds.drop('time_bnds') for ds in dslist_p]
        ds_c = xr.concat(dslist_c,pd.Index(outids,name='run')).chunk({'run':-1})
        ds_p = xr.concat(dslist_p,pd.Index(outids,name='run')).chunk({'run':-1})
        for dsi in dslist_c + dslist_p: dsi.close()

    if component=='ERF SW+LW': ds = ds_c - ds_p 
    else: ds = ds_p[['flux']] - ds_c[['flux']]
    ds = ds.rename({'flux':'rf'})

    for dsi in [ds_c,ds_p]: dsi.close()

    return ds 


########################################################################
#  END
########################################################################






