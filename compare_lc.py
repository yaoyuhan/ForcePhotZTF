#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:00:39 2018

@author: yuhanyao
"""
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


def plot_ztf_lc_marshal(lc, axi, jdcol = 'jdobs', filtercol = 'filter',
                        magcol = 'magpsf', emagcol='sigmamagpsf', nonvalue=99,
                        limmagcol = 'limmag', rstr = '"r"', gstr = '"g"',
                        istr = '"i"'):
    first_ix = np.where(lc[magcol].values!=nonvalue)[0][0]
    end_ix = np.where(lc[magcol].values!=nonvalue)[0][-1]
    
    first_date = lc[jdcol][first_ix]
    end_date = lc[jdcol][end_ix]
    
    start = np.where(lc[jdcol]>(first_date - 20))[0][0]
    end = np.where(lc[jdcol]<(end_date + 5))[0][-1]
    
    jdobs_ = lc[jdcol].values[start:end]
    filter_ = lc[filtercol].values[start:end]
    mag_ = lc[magcol].values[start:end]
    emag_ = lc[emagcol].values[start:end]
    limmag_ = lc[limmagcol].values[start:end]
    
    ind_r = np.all([mag_ != nonvalue, filter_ == rstr], axis=0)
    ind_g = np.all([mag_ != nonvalue, filter_ == gstr], axis=0)
    ind_i = np.all([mag_ != nonvalue, filter_ == istr], axis=0)
    
    maxr = 18.5
    minr = 18.5
    maxg = 18.5
    ming = 18.5
    maxi = 18.5
    mini = 18.5
    # limmaxr = 18.5
    # limmaxg = 18.5
    # limmaxi = 18.5
    
    if np.sum(ind_r)!=0:
        axi.errorbar(jdobs_[ind_r], mag_[ind_r], emag_[ind_r], fmt='.r')
        maxr = mag_[ind_r].max()
        minr = mag_[ind_r].min()
        
    if np.sum(ind_g)!=0:
        axi.errorbar(jdobs_[ind_g], mag_[ind_g], emag_[ind_g], fmt='.g')
        maxg = mag_[ind_g].max()
        ming = mag_[ind_g].min()
        
    if np.sum(ind_i)!=0:
        axi.errorbar(jdobs_[ind_i], mag_[ind_i], emag_[ind_i], fmt='.', color='orange')
        maxi = mag_[ind_i].max()
        mini = mag_[ind_i].min()
    
    ix_r = np.all([mag_ == nonvalue, filter_ == rstr], axis=0)
    ix_g = np.all([mag_ == nonvalue, filter_ == gstr], axis=0)
    ix_i = np.all([mag_ == nonvalue, filter_ == istr], axis=0)
    
    dy = 0.2
    if np.sum(ix_r)!=0:
        for j in range(np.sum(ix_r)):
            axi.arrow(jdobs_[ix_r][j], limmag_[ix_r][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc='salmon', ec='salmon', linewidth=1.5, 
                      linestyle='-.')
        # limmaxr = limmag_[ix_r].max()
            
    if np.sum(ix_g)!=0:
        for j in range(np.sum(ix_g)):
            axi.arrow(jdobs_[ix_g][j], limmag_[ix_g][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc='palegreen', ec='palegreen', linewidth=1.5, 
                      linestyle='-.')
        # limmaxg = limmag_[ix_g].max()
        
    if np.sum(ix_i)!=0:    
        for j in range(np.sum(ix_i)):
            axi.arrow(jdobs_[ix_i][j], limmag_[ix_i][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc='gold', ec='gold', linewidth=1.5, 
                      linestyle='-.')
        # limmaxi = limmag_[ix_i].max()
    
    axi.set_xlim(jdobs_[0]-5, jdobs_[-1]+5)
        
    ymax = max(maxr, maxg, maxi
               #, limmaxr, limmaxg, limmaxi
               )
    ymin = min(minr, ming, mini#, 
               #limmag_[ix_r].min()-dy, limmag_[ix_g].min()-dy
               )
    axi.set_ylim(ymax+0.1, ymin-0.1)
    
    
def plot_ztf_lc_force(lc, axi, jdcol = 'jdobs', filtercol = 'filter',
                      magcol = 'magpsf', emagcol='e_magpsf', nonvalue=99,
                      limmagcol = 'limmag_r=0.3', rstr = 'r', gstr = 'g',
                      istr = 'i', r_thre= 0.3):
    mag_ = deepcopy(lc[magcol].values)
    rs = lc['r_value'].values
    ix = rs > r_thre
    mag_[~ix] = nonvalue
    first_ix = np.where(mag_!=nonvalue)[0][0]
    end_ix = np.where(lc[magcol].values!=nonvalue)[0][-1]
    
    first_date = lc[jdcol][first_ix]
    end_date = lc[jdcol][end_ix]
    
    start = np.where(lc[jdcol]>(first_date - 20))[0][0]
    end = np.where(lc[jdcol]<(end_date + 5))[0][-1]
    
    jdobs_ = deepcopy(lc[jdcol].values[start:end])
    filter_ = deepcopy(lc[filtercol].values[start:end])
    mag_ = mag_[start:end]
    emag_ = lc[emagcol].values[start:end]
    limmag_ = lc[limmagcol].values[start:end]
    bad = deepcopy(lc['nbad'][start:end])
    ind_bad = bad!=0
    print ('%d epochs with bad pixels' %np.sum(ind_bad))
    
    ind_r = np.all([mag_ != nonvalue, filter_ == rstr], axis=0)
    ind_g = np.all([mag_ != nonvalue, filter_ == gstr], axis=0)
    ind_i = np.all([mag_ != nonvalue, filter_ == istr], axis=0)
    
    maxr = 18.5
    minr = 18.5
    maxg = 18.5
    ming = 18.5
    maxi = 18.5
    mini = 18.5
    # limmaxr = 18.5
    # limmaxg = 18.5
    # limmaxi = 18.5
    
    if np.sum(ind_r)!=0:
        ind_r_good = ind_r & (~ind_bad)
        ind_r_bad = ind_r & (ind_bad)
        print ('   %d bad pixels in r with detection'%np.sum(ind_r_bad))
        axi.errorbar(jdobs_[ind_r_good], mag_[ind_r_good], emag_[ind_r_good], fmt='.r')
        try:
            maxr = mag_[ind_r].max()
            minr = mag_[ind_r].min()
        except:
            1+1
        axi.errorbar(jdobs_[ind_r_bad], mag_[ind_r_bad], emag_[ind_r_bad], fmt='.',
                     color='k')
        
    if np.sum(ind_g)!=0:
        ind_g_good = ind_g & (~ind_bad)
        ind_g_bad = ind_g & (ind_bad)
        print ('   %d bad pixels in g with detection'%np.sum(ind_g_bad))
        axi.errorbar(jdobs_[ind_g_good], mag_[ind_g_good], emag_[ind_g_good], fmt='.g')
        try:
            maxg = mag_[ind_g].max()
            ming = mag_[ind_g].min()
        except:
            1+1
        axi.errorbar(jdobs_[ind_g_bad], mag_[ind_g_bad], emag_[ind_g_bad], fmt='.',
                     color='k')
        
    if np.sum(ind_i)!=0:
        ind_i_good = ind_i & (~ind_bad)
        ind_i_bad = ind_i & (ind_bad)
        print ('   %d bad pixels in i with detection'%np.sum(ind_i_bad))
        axi.errorbar(jdobs_[ind_i_good], mag_[ind_i_good], emag_[ind_i_good], fmt='.', color='orange')
        axi.errorbar(jdobs_[ind_i_bad], mag_[ind_i_bad], emag_[ind_i_bad], fmt='.', color='k')
        try:
            maxi = mag_[ind_i_good].max()
            mini = mag_[ind_i_good].min()
        except:
            1+1
    
    ix_r = np.all([mag_ == nonvalue, filter_ == rstr], axis=0)
    ix_g = np.all([mag_ == nonvalue, filter_ == gstr], axis=0)
    ix_i = np.all([mag_ == nonvalue, filter_ == istr], axis=0)
    
    dy = 0.2
    if np.sum(ix_r)!=0:
        subbad = ind_bad[ix_r].values
        print ('   %d bad pixels in r without detection'%np.sum(subbad))
        for j in range(len(subbad)):
            if subbad[j] ==0:
                fc = 'salmon'
                ec = 'salmon'
            else:
                fc = 'gray'
                ec = 'gray'
            axi.arrow(jdobs_[ix_r][j], limmag_[ix_r][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc=fc, ec=ec, linewidth=1.5, 
                      linestyle='-.')
        # limmaxr = limmag_[ix_r].max()
            
    if np.sum(ix_g)!=0:
        subbad = ind_bad[ix_g].values
        print ('   %d bad pixels in g without detection'%np.sum(subbad))
        for j in range(len(subbad)):
            if subbad[j] ==0:
                fc = 'palegreen'
                ec = 'palegreen'
            else:
                fc = 'gray'
                ec = 'gray'
            axi.arrow(jdobs_[ix_g][j], limmag_[ix_g][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc=fc, ec=ec, linewidth=1.5, 
                      linestyle='-.')
        # limmaxg = limmag_[ix_g].max()
        
    if np.sum(ix_i)!=0:    
        subbad = ind_bad[ix_i].values
        print ('   %d bad pixels in i without detection'%np.sum(subbad))
        for j in range(len(subbad)):
            if subbad[j] ==0:
                fc = 'gold'
                ec = 'gold'
            else:
                fc = 'gray'
                ec = 'gray'
            axi.arrow(jdobs_[ix_i][j], limmag_[ix_i][j]-0.2, 
                      head_width=0.4, head_length=0.1,
                      dx = 0, dy = dy, fc=fc, ec=ec, linewidth=1.5, 
                      linestyle='-.')
        # limmaxi = limmag_[ix_i].max()
    
    axi.set_xlim(jdobs_[0]-5, jdobs_[-1]+5)
        
    ymax = max(maxr, maxg, maxi
               #, limmaxr, limmaxg, limmaxi
               )
    ymin = min(minr, ming, mini#, 
               #limmag_[ix_r].min()-dy, limmag_[ix_g].min()-dy
               )
    axi.set_ylim(ymax+0.1, ymin-0.1)
    
    
def plot_ztf_lc_force_flux(lc, axi, jdcol = 'jdobs', filtercol = 'filter',
                           fluxcol = 'flux/flux0', efluxcol='e_flux/flux0', 
                           nonvalue=99, magcol = 'magpsf',
                           limmagcol = 'limmag_r=0.3', rstr = 'r', gstr = 'g',
                           istr = 'i', r_thre= 0.3):
    rs = lc['r_value'].values
    ix = rs > r_thre
    mag_ = deepcopy(lc[magcol].values)
    mag_[~ix] = nonvalue
    first_ix = np.where(mag_!=nonvalue)[0][0]
    end_ix = np.where(lc[magcol].values!=nonvalue)[0][-1]
    
    first_date = lc[jdcol][first_ix]
    end_date = lc[jdcol][end_ix]
    
    start = np.where(lc[jdcol]>(first_date - 20))[0][0]
    end = np.where(lc[jdcol]<(end_date + 5))[0][-1]
    
    jdobs_ = deepcopy(lc[jdcol].values[start:end])
    filter_ = deepcopy(lc[filtercol].values[start:end])
    flux_ = deepcopy(lc[fluxcol].values[start:end])
    eflux_ = lc[efluxcol].values[start:end]
    bad = deepcopy(lc['nbad'].values)[start:end]
    ind_bad = bad!=0
    print ('%d epochs with bad pixels!' %np.sum(ind_bad))
    
    ind_r = filter_ == rstr
    ind_g = filter_ == gstr
    ind_i = filter_ == istr
    
    if np.sum(ind_r)!=0:
        ind_r_good = ind_r & (~ind_bad)
        ind_r_bad = ind_r & (ind_bad)
        print ('   %d bad pixels in r with detection'%np.sum(ind_r_bad))
        axi.errorbar(jdobs_[ind_r_bad], flux_[ind_r_bad], eflux_[ind_r_bad], fmt='.',
                     color='k', alpha=0.5)
        axi.errorbar(jdobs_[ind_r_good], flux_[ind_r_good], eflux_[ind_r_good], fmt='.r')
        
        
    if np.sum(ind_g)!=0:
        ind_g_good = ind_g & (~ind_bad)
        ind_g_bad = ind_g & (ind_bad)
        print ('   %d bad pixels in g with detection'%np.sum(ind_g_bad))
        axi.errorbar(jdobs_[ind_g_bad], flux_[ind_g_bad], eflux_[ind_g_bad], fmt='.',
                     color='k', alpha=0.5)
        axi.errorbar(jdobs_[ind_g_good], flux_[ind_g_good], eflux_[ind_g_good], fmt='.g')
        
        
    if np.sum(ind_i)!=0:
        ind_i_good = ind_i & (~ind_bad)
        ind_i_bad = ind_i & (ind_bad)
        print ('   %d bad pixels in i with detection'%np.sum(ind_i_bad))
        axi.errorbar(jdobs_[ind_i_bad], flux_[ind_i_bad], eflux_[ind_i_bad], fmt='.', 
                     color='k', alpha=0.5)
        axi.errorbar(jdobs_[ind_i_good], flux_[ind_i_good], eflux_[ind_i_good], fmt='.', color='orange')
    

    axi.set_xlim(jdobs_[0]-5, jdobs_[-1]+5)

    
def compare_forcephot_marshal(name, targetdir, r_thre = 0.3, 
                              xmax=None, xmin = None,
                              ymin1 = None, ymax1 = None,
                              ymin2 = None, ymax2 = None):

    marshal_lc = pd.read_csv(targetdir+'lightcurves/marshal_lightcurve_'+name+'.csv')    
    force_lc = pd.read_csv(targetdir+'lightcurves/force_phot_'+name+'.csv')
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    matplotlib.rcParams.update({'font.size': 15})
    plot_ztf_lc_marshal(marshal_lc, axi=ax[0], jdcol = 'jdobs', filtercol = 'filter',
                        magcol = 'magpsf', emagcol='sigmamagpsf', nonvalue=99,
                        limmagcol = 'limmag', rstr = '"r"', gstr = '"g"',
                        istr = '"i"')
    
    limmagcol = 'limmag_r=' + repr(r_thre)
    plot_ztf_lc_force(lc = force_lc, axi=ax[1], jdcol = 'jdobs', filtercol = 'filter',
                      magcol = 'magpsf', emagcol='e_magpsf', nonvalue=99,
                      limmagcol = limmagcol, rstr = 'r', gstr = 'g',
                      istr = 'i', r_thre= r_thre)
    
    plot_ztf_lc_force_flux(lc = force_lc, axi=ax[2], jdcol = 'jdobs', filtercol = 'filter',
                           fluxcol = 'Fpsf/F0', efluxcol='e_Fpsf/F0',
                           magcol = 'magpsf', nonvalue=99, limmagcol = limmagcol, 
                           rstr = 'r', gstr = 'g', istr = 'i', r_thre= r_thre)
    
    xlim0 = ax[0].get_xlim()
    xlim1 = ax[1].get_xlim()
    xmin_ = min(xlim0[0], xlim1[0])
    xmax_ = max(xlim0[1], xlim1[1])  
    if xmax==None:
        xmax = xmax_
    if xmin==None:
        xmin = xmin_
    ax[0].set_xlim(xmin, xmax)
    ax[1].set_xlim(xmin, xmax)
    ax[2].set_xlim(xmin, xmax)
    
    
    ylim0 = ax[0].get_ylim()
    ylim1 = ax[1].get_ylim()
    ymin1_ = min(ylim0[1], ylim1[1])
    ymax1_ = max(ylim0[0], ylim1[0])
    if ymin1 == None:
        ymin1 = ymin1_
    if ymax1 == None:
        ymax1 = ymax1_
    ax[0].set_ylim(ymax1, ymin1)
    ax[1].set_ylim(ymax1, ymin1)
    ax[0].text(xmin+0.8*(xmax-xmin), ymax1-0.9*(ymax1-ymin1), 'from Marshal', color='blue')
    ax[1].text(xmin+0.8*(xmax-xmin), ymax1-0.9*(ymax1-ymin1), 'force photometry', color='blue')
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xlabel('JD')
    ax[0].set_ylabel('mag')
    ax[1].set_ylabel('mag')
    ax[2].set_ylabel(r'$f/f_0$')
    
    ymin2_, ymax2_ = ax[2].get_ylim()
    if ymin2 == None:
        ymin2 = ymin2_
    if ymax2 == None:
        ymax2 = ymax2_
    ax[2].set_ylim(ymin2, ymax2)
    ax[2].text(xmin+0.8*(xmax-xmin), ymin2+0.9*(ymax2-ymin2), 'in flux unit', color='blue')
    ax[0].tick_params(axis = 'both', which='both', direction='in')
    ax[1].tick_params(axis = 'both', which='both', direction='in')
    ax[0].set_title(name)
    
    plt.tight_layout(w_pad = 0.1)
    ax[0].grid(linestyle=':')
    ax[1].grid(linestyle=':')
    ax[2].grid(linestyle=':')
    plt.savefig(targetdir+'/compare_lc_'+name+'_psf.pdf')
    
    
    # plt.plot(yao_lc['jdobs']-2458300, yao_lc['magpsf'], '.')
    # plt.ylim(27, 15)
    
    
    
    
    
    