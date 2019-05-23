#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:26:28 2018

@author: yuhanyao
"""
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size']=14

import astropy.io.ascii as asci
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from ForcePhotZTF.keypairs import get_keypairs

DEFAULT_AUTHs = get_keypairs()
DEFAULT_AUTH_ipac = DEFAULT_AUTHs[2]


def read_ipac_lc(name, targetdir):
    myfile = targetdir + "lightcurves/forcedphotometry_ipac_lc.txt"
    f = open(myfile)
    lines = f.readlines()
    f.close()
    tb = asci.read(lines[69:])
    colnames = (lines[67][1:].split('\n'))[0].split(', ')
    for j in range(len(colnames)):
        tb.rename_column('col%d'%(j+1), colnames[j])   
    if tb['forcediffimfluxunc'].dtype in ['<U16', '<U17', '<U18', '<U19']:
        ix = tb['forcediffimfluxunc']=='null'
        tb = tb[~ix]
        tb['forcediffimfluxunc'] = np.array(tb['forcediffimfluxunc'], dtype=float)
    tb['forcediffimflux'] = np.array(tb['forcediffimflux'], dtype=float)
    tb = tb.to_pandas()
    
    tb.rename(columns={'forcediffimflux':'Fpsf',
                       'forcediffimfluxunc':'Fpsf_unc',
                       'zpdiff':'zp',
                       'zpmaginpsciunc':'ezp',
                       'jd':'jdobs',
                       'forcediffimchisq':'chi2_red',
                       'sciinpseeing':'seeing'}, inplace=True)
    
    #ix = tb['programid']==2
    #tb = tb[ix]
    
    F0 = 10**(tb['zp'].values/2.5)
    eF0 = F0 / 2.5 * np.log(10) * tb['ezp'].values
    Fpsf = tb['Fpsf'].values
    eFpsf = tb['Fpsf_unc'].values
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    tb['Fratio'] = Fratio
    tb['Fratio_unc'] = eFratio
    
    filt = tb['filter']
    filterid = np.zeros(len(tb))
    filterid[filt=='ZTF_g']=1
    filterid[filt=='ZTF_r']=2
    filterid[filt=='ZTF_i']=3
    tb['filterid'] = filterid
    
    tb['chi2_red'] = np.array([np.float(x) for x in tb['chi2_red'].values])
    
    tb['fcqfid'] = tb['field']*10000 + tb['ccdid']*100 + tb['qid']*10 + tb['filterid']
    
    tb['diffimgname'] = [x[34:-3] for x in tb['diffilename'].values]
    return tb


def read_mcmc_lc(name, targetdir):
    info_file = targetdir+'lightcurves/force_phot_{}_info_ref.fits'.format(name)
    # xy_file = targetdir+'lightcurves/xydata_{}.fits'.format(name)
    
    info_tbl = Table.read(info_file)
    # xy_tbl = Table.read(xy_file)
    info_df = info_tbl.to_pandas()
    # xy_df = xy_tbl.to_pandas()

    mylc = pd.read_hdf(targetdir+'lightcurves/'+name+'_force_phot_nob.h5')
    mylc['jdref_start'] = info_df['jdref_start'].values
    mylc['jdref_end'] = info_df['jdref_end'].values

    mylc['fcqfid'] = mylc['fieldid']*10000 + mylc['ccdid']*100 + mylc['qid']*10 + mylc['filterid']
    tb = mylc
    F0 = 10**(tb['zp'].values/2.5)
    eF0 = F0 / 2.5 * np.log(10) * tb['ezp'].values
    Fpsf = tb['Fmcmc'].values
    eFpsf = tb['Fmcmc_unc'].values
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    tb['Fratio'] = Fratio
    tb['Fratio_unc'] = eFratio
    
    tb['diffimgname'] = [x.decode("utf-8") for x in tb['diffimgname'].values]
    return tb
    
    

def plotlcs(tb, name, targetdir, seeing_cut = 7.):
    tb = tb[tb.seeing<seeing_cut]
    
    fcqfs = np.unique(tb['fcqfid'].values)
    
    colors_g = ['limegreen', 'c', 'skyblue']
    colors_r = ['r', 'm', 'pink']
    colors_i = ['gold', 'orange', 'y']
        
    plt.figure(figsize=(12,6))
    for fcqfid in fcqfs:
        ix = tb['fcqfid'].values==fcqfid
        if fcqfid % 10 ==1:
            color=colors_g[0]
            colors_g = colors_g[1:]
        elif fcqfid % 10 == 2:
            color=colors_r[0]
            colors_r = colors_r[1:]
        else:
            color=colors_i[0]
            colors_i = colors_i[1:]
        thistime = (tb['jdobs'][ix] - 2458000)
        plt.errorbar(thistime, tb['Fratio'][ix], tb['Fratio_unc'][ix], 
                     fmt='.', color=color, label = 'fcqfid = %d, Nobs = %d'%(fcqfid, np.sum(ix)))
        
    ax = plt.gca()
    ylims1 = ax.get_ylim()
        
    plt.ylim(ylims1[0], ylims1[1])
    plt.grid(ls=":")
    plt.xlabel('JD - 2458000 (days)')
    plt.ylabel('f/f0')
    plt.legend(loc = 'best')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(targetdir + name+'_fig_lc'+'.pdf')
    
    

def get_recerence_jds(name, targetdir, only_partnership=False, retain_iband = True,
                      oldsuffix = '_info.fits', newsuffix = '_info_ref.fits', verbose=True):
    print ('Start getting jd of reference exposures for %s'%name)
    s = requests.Session()
    
    s.post('https://irsa.ipac.caltech.edu/account/signon/login.do?josso_cmd=login', 
           data={'josso_username': DEFAULT_AUTH_ipac[0], 'josso_password': DEFAULT_AUTH_ipac[1]})
    
    mylc = Table(fits.open(targetdir+'lightcurves/force_phot_' + name + oldsuffix)[1].data)
    
    if retain_iband==False:
        ix= np.any([mylc['filter']=='r', mylc['filter']=='g'], axis=0)
        mylc = mylc[ix]
    
    if only_partnership==True:
        ix = mylc['programid']==2
        mylc = mylc[ix]
        
    if np.sum(mylc['qid']==99)!=0:
        if verbose==True:
            print ('reassign qid based on file name for %s'%(name))
        mylc['qid'] = [np.int(x.split('_o_q')[1].split('_')[0]) for x in mylc['diffimgname']]
        
    mylc['fcqf'] = mylc['fieldid']*10000 + mylc['ccdid']*100 + mylc['qid']*10 + mylc['filterid']
    fcq_uniq = []
    for x in mylc['fcqf']:
        if x not in fcq_uniq:
            fcq_uniq.append(x)
    fcq_uniq = np.array(fcq_uniq)
        
    jdref_start = np.zeros(len(mylc))
    jdref_end = np.zeros(len(mylc))
        
    for j in range(len(fcq_uniq)):
        fcqnow = fcq_uniq[j]
        temp1 = fcqnow - fcqnow%10000
        fieldnow = np.int(temp1/10000)
        temp2 = fcqnow - temp1
        temp3 = temp2 - temp2%100
        ccdidnow = np.int(temp3/100)
        temp4 = temp2 - temp3
        qidnow = np.int((temp4 - temp4%10)/10)
        filteridnow = temp4 - qidnow*10
        if filteridnow==1:
            fltidnow = 'zg'
        elif filteridnow==2:
            fltidnow = 'zr'
            
        # 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE=field=824%20AND%20ccdid=15%20AND%20qid=1%20AND%20filtercode=%271%27&CT=csv' 
        url = 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE=field=' +\
                '%d'%(fieldnow)+'%20AND%20ccdid='+'%d'%(ccdidnow) +\
                '%20AND%20qid='+'%d'%(qidnow)+\
                '%20AND%20filtercode=%27'+'%s'%(fltidnow)+'%27' 
        r = requests.get(url, cookies=s.cookies)
        stringnow = r.content
        stnow = stringnow.decode("utf-8")
        tbnowj = asci.read(stnow)
        t0 = tbnowj['startobsdate'].data[0]
        t1 = tbnowj['endobsdate'].data[0]
        tstart = Time(t0.split(' ')[0] + 'T' + t0.split(' ')[1][:-3], 
                      format='isot', scale='utc')
        tend = Time(t1.split(' ')[0] + 'T' + t1.split(' ')[1][:-3], 
                    format='isot', scale='utc')
        
        ind = mylc['fcqf'] == fcqnow
        jdref_start[ind] = tstart.jd
        jdref_end[ind] = tend.jd
        if verbose==True:
            print ('fieldid: %d, ccdid: %d, qid: %d, filterid: %d \n \t startjd: %.2f, endjd: %.2f \n'
                   %(fieldnow, ccdidnow, qidnow, filteridnow, tstart.jd, tend.jd))
            
    mylc['jdref_start'] = jdref_start
    mylc['jdref_end'] = jdref_end
        
    mylc.write(targetdir+'lightcurves/force_phot_' + name + newsuffix, overwrite=True)  
    
    
    
def get_recerence_jds_simple(filein, fileout, only_partnership=False, retain_iband = True, verbose=True):
    s = requests.Session()
    
    s.post('https://irsa.ipac.caltech.edu/account/signon/login.do?josso_cmd=login', 
           data={'josso_username': DEFAULT_AUTH_ipac[0], 'josso_password': DEFAULT_AUTH_ipac[1]})
    
    mylc = Table(fits.open(filein)[1].data)
    
    if retain_iband==False:
        ix= np.any([mylc['filter']=='r', mylc['filter']=='g'], axis=0)
        mylc = mylc[ix]
    
    if only_partnership==True:
        ix = mylc['programid']==2
        mylc = mylc[ix]
        
    if np.sum(mylc['qid']==99)!=0:
        if verbose==True:
            print ('reassign qid based on file name for this')
        mylc['qid'] = [np.int(x.split('_o_q')[1].split('_')[0]) for x in mylc['diffimgname']]
        
    mylc['fcqf'] = mylc['fieldid']*10000 + mylc['ccdid']*100 + mylc['qid']*10 + mylc['filterid']
    fcq_uniq = []
    for x in mylc['fcqf']:
        if x not in fcq_uniq:
            fcq_uniq.append(x)
    fcq_uniq = np.array(fcq_uniq)
        
    jdref_start = np.zeros(len(mylc))
    jdref_end = np.zeros(len(mylc))
        
    for j in range(len(fcq_uniq)):
        fcqnow = fcq_uniq[j]
        temp1 = fcqnow - fcqnow%10000
        fieldnow = np.int(temp1/10000)
        temp2 = fcqnow - temp1
        temp3 = temp2 - temp2%100
        ccdidnow = np.int(temp3/100)
        temp4 = temp2 - temp3
        qidnow = np.int((temp4 - temp4%10)/10)
        filteridnow = temp4 - qidnow*10
        if filteridnow==1:
            fltidnow = 'zg'
        elif filteridnow==2:
            fltidnow = 'zr'
            
        # 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE=field=824%20AND%20ccdid=15%20AND%20qid=1%20AND%20filtercode=%271%27&CT=csv' 
        url = 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE=field=' +\
                '%d'%(fieldnow)+'%20AND%20ccdid='+'%d'%(ccdidnow) +\
                '%20AND%20qid='+'%d'%(qidnow)+\
                '%20AND%20filtercode=%27'+'%s'%(fltidnow)+'%27' 
        r = requests.get(url, cookies=s.cookies)
        stringnow = r.content
        stnow = stringnow.decode("utf-8")
        tbnowj = asci.read(stnow)
        t0 = tbnowj['startobsdate'].data[0]
        t1 = tbnowj['endobsdate'].data[0]
        tstart = Time(t0.split(' ')[0] + 'T' + t0.split(' ')[1][:-3], 
                      format='isot', scale='utc')
        tend = Time(t1.split(' ')[0] + 'T' + t1.split(' ')[1][:-3], 
                    format='isot', scale='utc')
        
        ind = mylc['fcqf'] == fcqnow
        jdref_start[ind] = tstart.jd
        jdref_end[ind] = tend.jd
        if verbose==True:
            print ('fieldid: %d, ccdid: %d, qid: %d, filterid: %d \n \t startjd: %.2f, endjd: %.2f \n'
                   %(fieldnow, ccdidnow, qidnow, filteridnow, tstart.jd, tend.jd))
            
    mylc['jdref_start'] = jdref_start
    mylc['jdref_end'] = jdref_end
        
    mylc.write(fileout, overwrite=True)  
    
    