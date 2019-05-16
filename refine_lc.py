#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:26:28 2018

@author: yuhanyao
"""
import requests
import numpy as np
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



def plotlcs(tb, name, targetdir, t0jd=2458600, 
            jdend = 2458900, base_end = 2458480, ylims1 =None, 
            seeing_cut = 7.):
    
    tb['fcqfid'] = tb['fieldid']*10000 + tb['ccdid']*100 + tb['qid']*10 + tb['filterid']
    
    F0 = 10**(tb['zp'].values/2.5)
    eF0 = F0 / 2.5 * np.log(10) * tb['ezp'].values
    Fpsf = tb['Fmcmc'].values
    eFpsf = tb['Fmcmc_unc'].values
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    tb['Fratio'] = Fratio
    tb['Fratio_unc'] = eFratio
    
    ix = tb['jdobs']<jdend
    tb = tb[ix]
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
        thistime = (tb['jdobs'][ix] - t0jd)
        plt.errorbar(thistime, tb['Fratio'][ix], tb['Fratio_unc'][ix], 
                     fmt='.', color=color, label = 'fcqfid = %d, Nobs = %d'%(fcqfid, np.sum(ix)))
        
    ax = plt.gca()
    ylims1 = ax.get_ylim()
        
    plt.ylim(ylims1[0], ylims1[1])
    plt.grid(ls=":")
    plt.xlabel('rest frame days relative to B max')
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
    
    