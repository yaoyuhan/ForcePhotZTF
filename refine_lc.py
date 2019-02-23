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

import astropy.io.ascii as asci
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from ForcePhotZTF.keypairs import get_keypairs

DEFAULT_AUTHs = get_keypairs()
DEFAULT_AUTH_ipac = DEFAULT_AUTHs[2]


def quicklook_lc(name, targetdir, eFratio_upper_cut = 1e-8, seeing_cut = 5):
    
    mylc = Table(fits.open(targetdir+'lightcurves/force_phot_'+name+'_output2.fits')[1].data) 
    ix = mylc['nbad']==0
    mylc = mylc[ix]
    
    F0 = 10**(mylc['zp']/2.5)
    eF0 = F0 / 2.5 * np.log(10) * mylc['ezp']
    Fpsf = mylc['Fpsf']
    eFpsf = mylc['eFpsf']
    Fratio = Fpsf / F0
    eFratio2 = (eFpsf / F0)**2 + (Fpsf * eF0 / F0**2)**2
    eFratio = np.sqrt(eFratio2)
    
    mylc['Fratio'] = Fratio
    mylc['eFratio'] = eFratio
    
    if np.isnan(seeing_cut) == False:
        ix = mylc['seeing']<seeing_cut
        mylc = mylc[ix]

    if np.isnan(eFratio_upper_cut) == False:
        ix = mylc['eFratio'] < eFratio_upper_cut
        mylc = mylc[ix]
        
    fcqf_uniq = []
    for x in mylc['fcqf']:
        if x not in fcqf_uniq:
            fcqf_uniq.append(x)
    fcqf_uniq = np.array(fcqf_uniq)
    

    plt.figure(figsize=(10,10))
    matplotlib.rcParams.update({'font.size': 15})
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    colors_g = ['g', 'b', 'c']
    colors_r = ['r', 'm', 'orange']
    for j in range(len(fcqf_uniq)):
        sublc = mylc[mylc['fcqf']==fcqf_uniq[j]]
        
        if fcqf_uniq[j]%2==0: # r abnd
            color = colors_r[0]
            colors_r = colors_r[1:]

        else:
            color = colors_g[0]
            colors_g = colors_g[1:]
        
        ax1.errorbar(sublc['jdobs'], sublc['Fratio'], sublc['eFratio'], fmt='.', color=color)
        ax2.errorbar(sublc['jdobs'], sublc['Fpsf'].data, sublc['eFpsf'], fmt='.', color=color)
        
        ix = sublc['Fratio']>3*sublc['eFratio']
        subsublc = sublc[ix]
        
        mag = -2.5 * np.log10(subsublc['Fratio'])
        emag = 2.5 / np.log(10) * subsublc['eFratio'] / subsublc['Fratio']
        ax3.errorbar(subsublc['jdobs'], mag, emag, fmt='.', color=color)
     
    ylims1 = ax1.get_ylim()
    ylims1 = [ylims1[0], ylims1[1]]
    ylims2 = ax2.get_ylim()
    ylims2 = [ylims2[0], ylims2[1]]
    ylims3 = ax3.get_ylim()
    ax3.set_ylim(ylims3[1], ylims3[0])
    
    xlims = ax1.set_xlim()
    ax3.set_xlim(xlims[0], xlims[1])
        
    ax1.grid(linestyle=':')
    ax2.grid(linestyle=':')
    ax3.grid(linestyle=':')
    ax1.set_title(name)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xlabel('jd')
    ax1.set_ylabel('f/f_0')
    ax2.set_ylabel('DN')
    ax3.set_ylabel('mag')
    ax1.tick_params(axis='both', which='both', direction='in')
    ax2.tick_params(axis='both', which='both', direction='in')
    ax3.tick_params(axis='both', which='both', direction='in')
    plt.tight_layout()
    plt.savefig(targetdir+name+'_quicklook.pdf')


def get_recerence_jds(name, targetdir, partnership=False):
    print ('Start getting jd of reference exposures for %s'%name)
    s = requests.Session()
    s.post('https://irsa.ipac.caltech.edu/account/signon/login.do?josso_cmd=login', 
           data={'josso_username': DEFAULT_AUTH_ipac[0], 'josso_password': DEFAULT_AUTH_ipac[1]})
    
    mylc = Table(fits.open(targetdir+'lightcurves/force_phot_' + name + '_output1.fits')[1].data)
    ix= np.any([mylc['filter']=='r', mylc['filter']=='g'], axis=0)
    mylc = mylc[ix]
        
    if partnership==True:
        ix = mylc['programid']==2
        mylc = mylc[ix]
    irsaq = pd.read_csv(targetdir+'irsafile.csv')
    irsa = Table([irsaq['field'].values, irsaq['ccdid'].values, irsaq['fid'].values,
                  irsaq['qid'].values, irsaq['obsjd'].values], 
                    names = ['field', 'ccdid', 'fid', 'qid', 'obsjd'])
    
    irsa_jds = np.round(irsa['obsjd'], 7)
    irsa_fids = irsa['fid'].data
    irsa_code = irsa_fids*10000000 + irsa_jds
    irsa['code'] = irsa_code
        
    mylc_jds = np.round(mylc['jdobs'], 7)
    mylc_fids = np.zeros(len(mylc), dtype=int)
    mylc_fids[mylc['filter']=='g']=1
    mylc_fids[mylc['filter']=='r']=2
    mylc_code = mylc_fids*10000000 + mylc_jds
    mylc['filterid'] = mylc_fids
    mylc['code'] = mylc_code
    
    ix = np.in1d(mylc_code, irsa_code)
    assert np.sum(ix)==len(mylc_code)
    ix= np.in1d(irsa_code, mylc_code)
    assert np.sum(ix)==len(mylc_code)
    
    ira = irsa[ix]
        
    ira = ira[np.argsort(ira['code'])]
    mylc = mylc[np.argsort(mylc['code'])]
    mylc['field'] = ira['field']
    mylc['ccdid'] = ira['ccdid']
    mylc['qid'] = ira['qid']
    
    mylc['fcqf'] = mylc['field']*10000 + mylc['ccdid']*100 + mylc['qid']*10 + mylc['filterid']
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
        t0 = tbnowj['startobsdate'].data.data[0]
        t1 = tbnowj['endobsdate'].data.data[0]
        tstart = Time(t0.split(' ')[0] + 'T' + t0.split(' ')[1][:-3], 
                      format='isot', scale='utc')
        tend = Time(t1.split(' ')[0] + 'T' + t1.split(' ')[1][:-3], 
                    format='isot', scale='utc')
        
        ind = mylc['fcqf'] == fcqnow
        jdref_start[ind] = tstart.jd
        jdref_end[ind] = tend.jd
        print ('fieldid: %d, ccdid: %d, qid: %d, filterid: %d \n \t startjd: %.2f, endjd: %.2f \n'
               %(fieldnow, ccdidnow, qidnow, filteridnow, tstart.jd, tend.jd))
            
    mylc['jdref_start'] = jdref_start
    mylc['jdref_end'] = jdref_end
        
    mylc.write(targetdir+'lightcurves/force_phot_' + name + '_output2.fits',
               overwrite=True)  