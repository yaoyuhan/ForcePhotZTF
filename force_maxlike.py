#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:24:56 2019

@author: yuhanyao
"""
import glob
import numpy as np
from astropy.table import Table
from ForcePhotZTF.phot_class import ZTFphot


def get_forced_phot_maaxlike(name, targetdir, ra, dec,
                             r_psf = 3, r_bkg_in = 10, r_bkg_out = 15, verbose = False):
    '''
    Parameters:
    -----------
    r_bkg_in, r_bkg_out: [float] -optional-
        Inner and outer radius of an annulus to estimate background noise, in 
        pixel unit. Note that P48 pixel scale is 1.012 arcsec per pixel
        The default values are recommended.
        
    Outputs:
    --------
    `targetdir/lightcurves/force_phot_ZTFname_info.fits: [file]
        observation keywords [before calibration]
    
    '''     
    ################  read the coordinate and make figure dir #################
    imgdir = targetdir+'images_refdiff/'
    psfdir = targetdir+'images_diffpsf/'
    
    #################### load psf and refdiff image files #####################
    imgfiles = np.array(glob.glob(imgdir+'*.fits'))
    psffiles = np.array(glob.glob(psfdir+'*.fits'))
    arg = np.argsort(imgfiles)
    imgfiles = imgfiles[arg]
    arg = np.argsort(psffiles)
    psffiles = psffiles[arg]
    n = len(imgfiles)
    
    jdobs_ = np.zeros(n) #
    filter_ = np.array(['']*n) #
    zp_ = np.ones(n)*(99) #
    ezp_ = np.ones(n)*(99) #
    seeing_ = np.ones(n)*(99) #
    gains_ = np.zeros(n) #
    
    programid_ = np.zeros(n)
    field_ = np.zeros(n)
    ccdid_ = np.zeros(n)
    qid_ = np.zeros(n)
    filtercode_ = np.zeros(n)
    
    moonra_ = np.zeros(n) #
    moondec_ = np.zeros(n) #
    moonillf_  = np.zeros(n) #
    moonphas_ = np.zeros(n) #
    airmass_ = np.zeros(n) #
    
    nbads_ = np.zeros(n) #
    nbadbkgs_ = np.zeros(n) #
    bkgstd_ = np.zeros(n) #
    bkgmed_ = np.zeros(n) #
    
    Fmaxlike = np.zeros(n) #
    eFmaxlike = np.zeros(n) #
    chi2_reds = np.zeros(n) #
    
    status_ = np.ones(n) #
    
    ####################### dalta analysis: light curve #######################
    print ('Start saving cutouts and observation info for %s...'%name)
    for i in range(n):
        if i%50==0:
            print ('In progress %s: %d in %d...' %(name, i, n))
        imgpath = imgfiles[i]
        psfpath = psffiles[i]
        pobj = ZTFphot(name, ra, dec, imgpath, psfpath, r_psf, 
                       r_bkg_in, r_bkg_out, verbose)
        zp_[i] = pobj.zp
        ezp_[i] = pobj.e_zp
        seeing_[i] = pobj.seeing
        jdobs_[i] = pobj.obsjd
        filter_[i] = pobj.filter
        gains_[i] = pobj.gain
        programid_[i] = pobj.programid
        field_[i] = pobj.fieldid
        ccdid_[i] = pobj.ccdid
        qid_[i] = pobj.qid
        filtercode_[i] = pobj.filterid
        moonra_[i] = pobj.moonra
        moondec_[i] = pobj.moondec
        moonillf_[i] = pobj.moonillf
        moonphas_[i] = pobj. moonphas
        airmass_[i] = pobj.airmass
        
        if pobj.status == False:
            status_[i] = 0
            continue
        
        pobj.load_source_cutout() 
        if pobj.status == False:
            status_[i] = 0
            continue
            
        pobj.load_bkg_cutout()
        pobj.get_scr_cor_fn()  
        pobj.fit_psf()
        
        nbads_[i] = pobj.nbad
        nbadbkgs_[i] = pobj.nbad_bkg
        bkgstd_[i] = pobj.bkgstd
        bkgmed_[i] = pobj.bkgmed
        
        Fmaxlike[i] = pobj.Fpsf
        eFmaxlike[i] = pobj.eFpsf
        chi2_reds[i] = pobj.chi2_red
        
        
    print ('\n')
    
    ####################### save the results to a file ########################
    diffimgname = np.array([x.split('/')[-1]for x in imgfiles])
    psfimgname = np.array([x.split('/')[-1]for x in psffiles])
    
    data = [jdobs_, filter_, seeing_, gains_, zp_, ezp_, 
            programid_, field_, ccdid_, qid_, filtercode_, 
            moonra_, moondec_, moonillf_, moonphas_, airmass_,
            nbads_, nbadbkgs_, bkgstd_,  bkgmed_, diffimgname, psfimgname,
            Fmaxlike, eFmaxlike. chi2_reds]
    
    my_lc = Table(data,names=['jdobs','filter', 'seeing', 'gain', 'zp', 'ezp',
                              'programid', 'fieldid', 'ccdid', 'qid', 'filterid', 
                              'moonra', 'moondec', 'moonillf', 'moonphase', 'airmass',
                              'nbad', 'nbadbkg', 'bkgstd', 'bkgmed', 'diffimgname', 
                              'psfimgname', 'Flux_maxlike', 'Flux_maxlike_unc', 'chi2_nu'])
    
    mylc = my_lc[status_==1]
    print ('writing data to database')
    mylc.write(targetdir+'/lightcurves/force_phot_'+name+'_maxlikelihood_lc.fits', overwrite=True)
