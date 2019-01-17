#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:00:39 2018

@author: yuhanyao
"""
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from copy import deepcopy
from astropy.table import Table

    
def plot_forcephot_lc(name, targetdir):

    force_lc = Table(fits.open(targetdir+'lightcurves/forced_'+name+'.fits')[1].data)
    ix = force_lc['mag']==99
    lcu = deepcopy(force_lc[ix])
    lcd = deepcopy(force_lc[~ix])
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    matplotlib.rcParams.update({'font.size': 15})
    
    ixrm = (lcd['force']==0)&(lcd['filter']=='r')
    ax[0].errorbar(lcd['jdobs'][ixrm], lcd['mag'][ixrm], lcd['emag'][ixrm], fmt='.m')
    
    ixrg = (lcd['force']==0)&(lcd['filter']=='g')
    ax[0].errorbar(lcd['jdobs'][ixrg], lcd['mag'][ixrg], lcd['emag'][ixrg], fmt='.c')
    
    ixrf = (lcd['force']==1)&(lcd['filter']=='r')
    ax[0].errorbar(lcd['jdobs'][ixrf], lcd['mag'][ixrf], lcd['emag'][ixrf], fmt='.r')
    
    ixgf = (lcd['force']==1)&(lcd['filter']=='g')
    ax[0].errorbar(lcd['jdobs'][ixgf], lcd['mag'][ixgf], lcd['emag'][ixgf], fmt='.g')
    
    ixru = lcu['filter'] == 'r'
    ax[0].plot(lcu['jdobs'][ixru], lcu['limmag'][ixru], 'v', color='salmon')
    
    ixgu = lcu['filter'] == 'g'
    ax[0].plot(lcu['jdobs'][ixgu], lcu['limmag'][ixgu], 'v', color='palegreen')
    
    ylim0s = ax[0].get_ylim()
    ax[0].set_ylim(ylim0s[1], ylim0s[0])
    
    ixrm0 = (force_lc['force']==0)&(force_lc['filter']=='r')
    ax[1].errorbar(force_lc['jdobs'][ixrm0], force_lc['Fratio'][ixrm0], 
                   force_lc['eFratio'][ixrm0], fmt='.m')
    
    ixgm0 = (force_lc['force']==0)&(force_lc['filter']=='g')
    ax[1].errorbar(force_lc['jdobs'][ixgm0], force_lc['Fratio'][ixgm0], 
                   force_lc['eFratio'][ixgm0], fmt='.c')
    
    ixrf0 = (force_lc['force']==1)&(force_lc['filter']=='r')
    ax[1].errorbar(force_lc['jdobs'][ixrf0], force_lc['Fratio'][ixrf0], 
                   force_lc['eFratio'][ixrf0], fmt='.r')
    
    ixgf0 = (force_lc['force']==1)&(force_lc['filter']=='g')
    ax[1].errorbar(force_lc['jdobs'][ixgf0], force_lc['Fratio'][ixgf0], 
                   force_lc['eFratio'][ixgf0], fmt='.g')
    
    ax[0].tick_params(axis = 'both', which='both', direction='in')
    ax[1].tick_params(axis = 'both', which='both', direction='in')
    ax[0].set_xticklabels([])
    ax[1].set_xlabel('JD')
    ax[0].set_ylabel('mag')
    ax[1].set_ylabel(r'$f/f_0$')
    ax[0].set_title(name, fontsize=15)
    
    plt.tight_layout(w_pad = 0.1)
    ax[0].grid(linestyle=':')
    ax[1].grid(linestyle=':')
    plt.savefig(targetdir+'/lc_'+name+'_psf.pdf')
    plt.close()
    
    
    
    
    
    