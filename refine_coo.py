#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:27:04 2018

@author: yuhanyao
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import units
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from ztfquery import marshal
from phot_class import ZTFphot
    

def get_pos(name):
    """
    Get position of target, borrow from
    https://github.com/annayqho/FinderChart/blob/master/ztf_finder.py
    """
    m = marshal.MarshalAccess()
    m.load_target_sources()
    coords = m.get_target_coordinates(name)
    ra = coords.ra.values[0]
    dec = coords.dec.values[0]
    return ra, dec


def astrometry_spread(name, raset, decset, ra, dec, ra1, dec1):
    radius = 0.1
    fontsize = 15
    c1 = SkyCoord(ra = ra, dec = dec, unit = 'degree')
    c2 = SkyCoord(ra = ra1, dec = dec1, unit = 'degree')
    coords = SkyCoord(ra = raset, dec = decset, unit='degree')
    
    xs = []
    ys = []
    for i in range(len(coords)):
        sep1 = c1.separation(coords[i]).arcsec
        pos1 = c1.position_angle(coords[i]).rad
        x1 = sep1 * np.cos(pos1)
        y1 = sep1 * np.sin(pos1)
        xs.append(x1)
        ys.append(y1)
    xs = np.array(xs)
    ys = np.array(ys)
    
    sep2 = c1.separation(c2).arcsec
    pos2 = c1.position_angle(c2).rad
    x2 = sep2 * np.cos(pos2)
    y2 = sep2 * np.sin(pos2)
    
    plt.figure(figsize = (8, 7.5))
    for i in range(len(xs)):
        plt.plot([0, xs[i]], [0, ys[i]], 'k-', alpha=0.1, linewidth = 2.5, zorder=1)
        plt.plot(xs[i], ys[i], 'k.', zorder=1)
    plt.plot(0, 0, 'k.', label = '%d obs around peak'%len(raset))
    plt.plot(0, 0, 'r.', label='new centroid', zorder=2)
    plt.xlabel(r'$\Delta$'+'RA (arcsec)')
    plt.ylabel(r'$\Delta$'+'DEC (arcsec)')
    plt.title(name)
    
    plt.plot([0,x2], [0, y2], 'm-', zorder=2)
    plt.plot(x2, y2, 'm.', label='marshal', zorder=2)
    
    phis = np.linspace(0, 2*np.pi, 1000)
    xxs = radius * np.cos(phis)
    yys = radius * np.sin(phis)
    plt.plot(xxs, yys, 'g--', label='radius = '+repr(radius)+' arcsec', zorder=2)
    ax = plt.gca()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xylim = max(abs(xlims[0]), abs(xlims[1]), abs(ylims[0]), abs(ylims[1]))
    plt.xlim(-1*xylim, xylim)
    plt.ylim(-1*xylim, xylim)
    plt.legend(loc = 'best', fontsize = fontsize)
    plt.tight_layout()
     
    
    
def get_refined_coord(name, ra1, dec1, bad_thre, targetdir, peak_jd,
                      ndays_before_peak, ndays_after_peak):
    try:
        ra,dec = np.loadtxt(targetdir+'/coo.reg')
    except:
        imgdir = targetdir+'/images_refdiff/'
        psfdir = targetdir+'/images_diffpsf/'
        imgfiles = np.array(glob.glob(imgdir+'*.fits'))
        arg = np.argsort(imgfiles)
        imgfiles = imgfiles[arg]
        psffiles = np.array(glob.glob(psfdir+'*.fits'))
        arg = np.argsort(psffiles)
        psffiles = psffiles[arg]
        
        n = len(imgfiles)
        jdobs_ = np.zeros(n)
        
        for i in range(n):
            if i%50 == 0:
                print ('Reading: %d in %d...' %(i, n))
            imgpath = imgfiles[i]
            hd = fits.open(imgpath)[1].header
            jdobs_[i] = hd['obsjd']
        
        ix = np.any([np.all([(jdobs_ - peak_jd) < ndays_after_peak,
                             (jdobs_ - peak_jd) >= 0], axis=0),
                     np.all([(peak_jd - jdobs_) < ndays_before_peak,
                             (peak_jd - jdobs_) >=0], axis=0)], axis=0)
        if np.sum(ix)==0:
            print ('no obs around peak! Please check for program id')
        print ('%d obs around peak to find the centroid' %np.sum(ix))
        imgfiles = imgfiles[ix]
        psffiles = psffiles[ix]
        
        raset = []
        decset = []
        nbadset = []
        for i in range(len(imgfiles)):
            if i%10==0:
                print ('re-center: %d in %d' %(i, len(imgfiles)))
            imgpath = imgfiles[i]
            psfpath = psffiles[i]
            pobj = ZTFphot(name, ra1, dec1, imgpath, psfpath, bad_thre)
            if pobj.status == False:
                continue
            pobj.load_source_cutout()
            pobj.find_optimal_coo()
            raset.append(pobj.ra_cor)
            decset.append(pobj.dec_cor)
            nbadset.append(pobj.nbad)
            
        raset = np.array(raset)
        decset = np.array(decset)   
        nbadset = np.array(nbadset)
        
        ix = nbadset == 0
        print ('remove %d images with bad pixels' %np.sum(~ix))
        raset = raset[ix]
        decset = decset[ix]
    
        ra = np.median(raset)
        dec = np.median(decset)
        # plt.plot(raset, decset, 'k.')
        #  plt.plot(ra, dec, 'r.')
        cset = SkyCoord(raset*units.deg, decset*units.deg)
        centroid = SkyCoord(ra*units.deg, dec*units.deg)
        
        sep_arcsec = centroid.separation(cset).arcsec
        clip_sep = sigma_clip(sep_arcsec)
        while np.sum(clip_sep.mask)!=0:
            print ('remove %d because of >3 sigma deviation from others' %np.sum(clip_sep.mask))
            raset = raset[~clip_sep.mask]
            decset = decset[~clip_sep.mask]
            ra = np.median(raset)
            dec = np.median(decset)
            cset = SkyCoord(raset*units.deg, decset*units.deg)
            centroid = SkyCoord(ra*units.deg, dec*units.deg)
            sep_arcsec = centroid.separation(cset).arcsec
            clip_sep = sigma_clip(sep_arcsec)
        
        astrometry_spread(name, raset, decset, ra, dec, ra1, dec1)
        plt.savefig(targetdir+'/astrometry.pdf')
        plt.close()
        np.savetxt(targetdir+'/coo.reg', [ra,dec])
    return ra, dec
