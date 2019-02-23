#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:45:08 2018

@author: yuhanyao
"""
import os
import glob
import requests
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.io.ascii as asci
from astropy.table import Table

from ztfquery import query
from ztfquery.io import download_single_url

from ForcePhotZTF.keypairs import get_keypairs
from ForcePhotZTF.phot_class import ZTFphot
from ForcePhotZTF.refine_coo import get_refined_coord, get_pos

DEFAULT_AUTHs = get_keypairs()
DEFAULT_AUTH_marshal = DEFAULT_AUTHs[0]


def download_images_diffpsf_refdiff(targetdir, ra1, dec1, start_jd, 
                                    open_check = False):
    '''
    Download subtracted images and psf images from irsa
    '''
    subdir1 = os.path.dirname(targetdir+'/images_diffpsf/')
    try:
        os.stat(subdir1)
    except:
        os.mkdir(subdir1) 
    subdir2 = os.path.dirname(targetdir+'/images_refdiff/')
    try:
        os.stat(subdir2)
    except:
        os.mkdir(subdir2) 
        
    ############## Get metadata ocf all images at this location ################
    zquery = query.ZTFQuery()
    print ('\n')
    print("Querying for metadata...")
    if start_jd == None:
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.003)
    else:
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.003,
                             sql_query='obsjd>'+repr(start_jd))
    out = zquery.metatable
    final_out = out.sort_values(by=['obsjd'])
    final_out.to_csv(targetdir+'/irsafile.csv')
        
    urls, dl_loc = zquery.download_data(nodl=True)
    urls = np.array(urls)
    dl_loc = np.array(dl_loc)
    print ('Trying to download %d images from irsa...' %len(urls)) 
    
    for i in range(len(urls)):
        if i%50 == 0:
            print ('In progress: %d in %d' %(i,len(urls) ))
        _url = urls[i]
        _url_diffpsf = _url.split('sciimg')[0]+'diffimgpsf.fits'
        _url_ref = _url.split('sciimg')[0]+'scimrefdiffimg.fits.fz'
        fitsfile1 = _url_diffpsf.split('/')[-1]    
        fitsfile2 = _url_ref.split('/')[-1]    
        savename1 = subdir1+'/'+fitsfile1.split('.fits')[0]+'.fits'
        savename2 = subdir2+'/'+fitsfile2.split('.fits')[0]+'.fits'
            
        if os.path.isfile(savename1)==False:
            download_single_url(_url_diffpsf, fileout=savename1, cookies=None,
                                verbose=False)
        if os.path.isfile(savename2)==False:
            download_single_url(_url_ref, fileout=savename2, cookies=None,
                                verbose=False)
            
    ################## we do not have access to some images ###################
    original_names = []
    for i in range(len(urls)):
        original_names.append(urls[i].split('/')[-1].split('_sciimg.fits')[0])
    original_names = np.array(original_names)
    argo = np.argsort(original_names)
    original_names = original_names[argo]
        
    durls = glob.glob(subdir1+'/*.fits')
    downloaded_names = []
    for i in range(len(durls)):
        downloaded_names.append(durls[i].split('/')[-1].split('_diffimgpsf.fits')[0])
    downloaded_names = np.array(downloaded_names)
    argd = np.argsort(downloaded_names)
    downloaded_names = downloaded_names[argd]
        
    ix_why = np.in1d(original_names, downloaded_names)
    print ('%d images in %d we do not have data:' %(np.sum(~ix_why), len(urls)))
    print ('\n')

    ###################### check if files can be opened #######################
    if open_check == True:
        print ('checking if all files can be opened...')
        imgdir = targetdir+'/images_refdiff/'
        psfdir = targetdir+'/images_diffpsf/'
        imgfiles = np.array(glob.glob(imgdir+'*.fits'))
        arg = np.argsort(imgfiles)
        imgfiles = imgfiles[arg]
        psffiles = np.array(glob.glob(psfdir+'*.fits'))
        arg = np.argsort(psffiles)
        psffiles = psffiles[arg]
        n = len(imgfiles)
        
        if len(imgfiles)!=len(psffiles):
            imgstrings = [x.split('/')[-1][:-20] for x in imgfiles]
            psfstrings = [x.split('/')[-1][:-16] for x in psffiles]
            imgstrings = np.array(imgstrings)
            psfstrings = np.array(psfstrings)
            if len(imgstrings) > len(psfstrings):
                ix = np.in1d(imgstrings, psfstrings)
                for x in imgfiles[~ix]:
                    os.remove(x)
            else:
                ix = np.in1d(psfstrings, imgstrings)
                for x in psffiles[~ix]:
                    os.remove(x)
            imgdir = targetdir+'/images_refdiff/'
            psfdir = targetdir+'/images_diffpsf/'
            imgfiles = np.array(glob.glob(imgdir+'*.fits'))
            arg = np.argsort(imgfiles)
            imgfiles = imgfiles[arg]
            psffiles = np.array(glob.glob(psfdir+'*.fits'))
            arg = np.argsort(psffiles)
            psffiles = psffiles[arg]
            n = len(imgfiles)
        
        for i in range(n):
            if i%50 == 0:
                print ('In progress: %d in %d...' %(i, n))
            imgpath = imgfiles[i]
            psfpath = psffiles[i]
            try:
                fits.open(imgpath)[1].header
                fits.open(imgpath)[1].data
                fits.open(psfpath)[0].data
            except:
                print ('file broken, remove %s, %s' %(imgpath, psffiles[i]))
                os.remove(imgfiles[i])
                os.remove(psffiles[i])
        print ('\n')
    
    
def download_marshal_lightcurve(name, targetdir):
    '''
    download the marshal lightcurve and plot it, save the figure to targetdir
    '''
    try:
        os.stat(targetdir)
    except:
        os.mkdir(targetdir)
        
    try:
        os.stat(targetdir+'lightcurves/')
    except:
        os.mkdir(targetdir+'lightcurves/')
        
    ####################### download marshal lightcurve #######################
    try:
        os.stat(targetdir+'lightcurves/marshal_lc_'+name+'.csv')
    except:
        r = requests.get('http://skipper.caltech.edu:8080/cgi-bin/growth/plot_lc.cgi?name='+name, 
                         auth=(DEFAULT_AUTH_marshal[0], DEFAULT_AUTH_marshal[1]))
        tables = pd.read_html(r.content)
        mtb = tables[14]
        mtb = mtb.drop([0], axis=1)
        mtb.to_csv(targetdir+'lightcurves/'+'/marshal_lc_'+name+'.csv',header=False, index = False)
    
    ###################### visualize marshal lightcurve #######################
    mtb = asci.read(targetdir+'lightcurves/'+'/marshal_lc_'+name+'.csv')
    ix = mtb['absmag']!=99
    mtb = mtb[ix]
    
    ixr = mtb['filter']=='r'
    ixg = mtb['filter']=='g'
            
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(211)
        
    ax1.errorbar(mtb['jdobs'][ixr], mtb['mag'][ixr], mtb['emag'][ixr], fmt='.r', alpha=0.1)
    ax1.errorbar(mtb['jdobs'][ixg], mtb['mag'][ixg], mtb['emag'][ixg], fmt='.g', alpha=0.1)
    ax1.set_title(name, fontsize=14)
    ylim = ax1.get_ylim()
        
    isdiffops = [x=='True' for x in mtb['isdiffpos'].data]
    isdiffops = np.array(isdiffops)
    ixrr = isdiffops&ixr
    ixgg = isdiffops&ixg
    ax1.errorbar(mtb['jdobs'][ixrr], mtb['mag'][ixrr], mtb['emag'][ixrr], fmt='.r')
    ax1.errorbar(mtb['jdobs'][ixgg], mtb['mag'][ixgg], mtb['emag'][ixgg], fmt='.g')
    ax1.set_ylim(ylim[1], ylim[0])
    
    ix_peak = np.where(mtb['mag'][isdiffops]==min((mtb['mag'][isdiffops])))[0][0]
    marshal_peak_jd = mtb['jdobs'][isdiffops][ix_peak]
    plt.plot([marshal_peak_jd-10, marshal_peak_jd-10], [ylim[1], ylim[0]], 'k:')
    plt.plot([marshal_peak_jd+10, marshal_peak_jd+10], [ylim[1], ylim[0]], 'k:')
        
    ax2 = plt.subplot(212)
    if mtb['programid'].dtype=='<U4':
        ixno = mtb['programid']=='None'
        mtb = mtb[~ixno]
        ix = np.any([mtb['programid']=='0', mtb['programid']=='2', 
                     mtb['programid']=='3'], axis=0)
        mtb = mtb[ix]
    else:
        ix = np.any([mtb['programid']==0, mtb['programid']==2, 
                     mtb['programid']==3], axis=0)
        mtb = mtb[ix]
        
    ixr = mtb['filter']=='r'
    ixg = mtb['filter']=='g'
        
    isdiffops = [x=='True' for x in mtb['isdiffpos'].data]
    isdiffops = np.array(isdiffops)
    if len(isdiffops)>0:
        ixrr = isdiffops&ixr
        ixgg = isdiffops&ixg
        ax2.errorbar(mtb['jdobs'][ixr], mtb['mag'][ixr], mtb['emag'][ixr], fmt='.r', alpha=0.1)
        ax2.errorbar(mtb['jdobs'][ixg], mtb['mag'][ixg], mtb['emag'][ixg], fmt='.g', alpha=0.1)
        ax2.errorbar(mtb['jdobs'][ixrr], mtb['mag'][ixrr], mtb['emag'][ixrr], fmt='.r')
        ax2.errorbar(mtb['jdobs'][ixgg], mtb['mag'][ixgg], mtb['emag'][ixgg], fmt='.g')
    
        plt.plot([marshal_peak_jd-10, marshal_peak_jd-10], [ylim[1], ylim[0]], 'k:')
        plt.plot([marshal_peak_jd+10, marshal_peak_jd+10], [ylim[1], ylim[0]], 'k:')
            
    ax2.set_ylim(ylim[1], ylim[0])
    xlim = ax1.get_xlim()
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.set_xticklabels([])
    plt.tight_layout()
    plt.savefig(targetdir+'marshal_lc_plot'+name+'.pdf')
    plt.close()   
    

def prepare_forced_phot(name, targetdir = 'default',
                        before_marshal_detection = None,
                        detection_jd = None, recenter_coo = True,
                        ndays_before_peak = 8, ndays_after_peak = 10,
                        open_check = False):
    '''
    Preparation for force psf photometry light curve for ZTF a source:
        1) download images
        2) re-center the coordinate 
           (usually only small offset from the marshal, ~0.1 arcsec)
    
    Parameters:
    -----------
    name: [str]
        Name of a target on the marshal.
        
    targetdir: [str] -optional-
        Directory where the data should be stored. 
        - `dirout='default'`: All the data will be saved in target location 
                              (`currentdir/name/`)
                              
    before_marshal_detection: [None | float, >0] -optional-
        The number of days before the first marshal detection to download 
        archival images. A target may have lots of images well before the first
        detection. 
        - `before_marshal_detection=None`: download all archival images
        
    detection_jd: [float] -optional-
        The julian date when this target is first detected on Marshal
        Only useful when before_marshal_detection is not None
         - `detection_jd=None`: then this function will find this jd automatically
        
    recenter_coo: [bool] -optional-
        Re-determine the coordinate of the target by taking the median of all
        target positions within 
        [jd_peak - ndays_before_peak, jd_peak + ndays_after_peak]
        - `recenter_coo=False`: use the coordinate on Marshal webpage
    
    ndays_before_peak, ndays_after_peak: [int] -optional-
        see recenter_coo; Only useful when recenter_coo is True
        
    open_check: [bool] -optional-
        If true, each file will be check to see if it can be opened after download.
        Broken files seldom exist.
        
    Returns:
    -------
    None
    
    Outputs:
    --------
    `targetdir/lightcurves/marshal_lightcurve_ZTFname.csv`: [file]
        marshal lightcurve
        
    `targetdir/images_refdiff/`: [directory]
        contains difference images on which PSF photometry is performed
        
    `targetdir/images_diffpsf/`: [directory]
        contains images with psf models on each epoch
        
    `targetdir/irsafile.csv` & `targetdir/missingdata.txt`: [file]
        by-products in the process of downloading from irsa
        
    `targetdir/coo_marshal.reg`: [file]
        marshal coordinate of the source
    
    `targetdir/coo.reg`: [file]
        re-centered coordinate; only exist when recenter_coo == True
        
    `targetdir/astrometry.pdf`: [file]
        a figure showing how far the recentered coordinate is from the marshal
        coordinate; only exist when recenter_coo == True
    '''
         
    name = str(name)
    bad_threshold = -500 # Pixels with value < bad_threshold will be counted as bad pixels
    
    ##### make a directory to store all the data relavant to this target ######
    if targetdir == 'default':
        cwd = os.getcwd()
        targetdir = cwd+'/'+name+'/'
        try:
            os.stat(targetdir)
        except:
            os.mkdir(targetdir)
    else:
        if not targetdir.endswith(os.path.sep):
            targetdir += os.path.sep
        if not os.path.isdir(targetdir):
            os.mkdir(targetdir)
    
    ################ get the marshal coordinate of this target ################
    try:
        ra1, dec1 = np.loadtxt(targetdir+'/coo_marshal.reg')
    except:
        print ('\n')
        print ("Start getting coordinate for %s" %name)
        ra1, dec1 = get_pos(name)
        print ('%s: ra=%f, dec=%f'%(name, ra1, dec1))
        np.savetxt(targetdir+'/coo_marshal.reg', [ra1, dec1])
        
    ######################## download images from irsa ########################
    if before_marshal_detection == None:
        start_jd = None
    else:
        if detection_jd == None:
            marshal_lc = pd.read_csv(targetdir+'lightcurves/'+ \
                                     'marshal_lightcurve_'+name+'.csv')    
            detection_ix = np.where(marshal_lc['magpsf'].values!=99)[0][0]
            detection_jd = marshal_lc['jdobs'][detection_ix]
            
        start_jd = detection_jd - before_marshal_detection
    download_images_diffpsf_refdiff(targetdir, ra1, dec1, start_jd, open_check)
        
    ################# re-center the coordinate of the source ##################
    if recenter_coo == True:
        print ('Determining the coordinate based on observations around peak...')
        marshal_lc = pd.read_csv(targetdir+'lightcurves/'+ \
                                 'marshal_lc_'+name+'.csv')  
        mags = marshal_lc['mag'].values
        peak_ix = np.where(mags == mags.min())[0][0]
        peak_jd = marshal_lc['jdobs'][peak_ix]
        fade_ix = np.where(marshal_lc['mag'].values!=99)[0][-1]
        fade_jd = marshal_lc['jdobs'][fade_ix]
        if detection_jd == None:
            detection_ix = np.where(marshal_lc['mag'].values!=99)[0][0]
            detection_jd = marshal_lc['jdobs'][detection_ix]
        
        if (peak_jd - detection_jd) < ndays_before_peak:
            ndays_before_peak = peak_jd - detection_jd
            print ('setting ndays_before_peak to %.2f'%ndays_before_peak)
        if (fade_jd - peak_jd) < ndays_after_peak:
            ndays_after_peak = fade_jd - peak_jd
            print ('setting ndays_after_peak to %.2f' %ndays_after_peak)
        
        get_refined_coord(name, ra1, dec1, bad_threshold, targetdir, peak_jd,
                          ndays_before_peak, ndays_after_peak)
        
        
def get_force_lightcurve(name, targetdir, 
                         r_psf = 3, r_bkg_in = 25, r_bkg_out = 30,
                         plot_mod = 10000, manual_mask = False, 
                         col_mask_start = 0, col_mask_end = 0,                  
                         row_mask_start = 0, row_mask_end = 0, verbose = False):
    '''
    Parameters:
    -----------
    manual_mask: [bool] -optional-
        manually mask some pixels within the 25*25 cutout at every epoch. 
        If true, masked region will be a square of
        [col_mask_start:col_mask_end, row_mask_start:row_mask_end]
        currently only support square mask.
        Set manual_mask = True when there is bad subtraction at the same region
        on every image (e.g., nucleus of the host galaxy)
        
    plot_mod: [int] -optional-
        Visualization of the psf fitting every plot_mod images
        All figures will be saved to `targetdir/figures/`
        - `plot_mod=10000`: do not visualize the fitting
        - `plot_mod=1`: visualize every image 
    
    col_mask_start, col_mask_end, row_mask_start, row_mask_end: [int, >=0, <25] -optional-
        see manual_mask; Only useful when manual_mask = True
        
    r_bkg_in, r_bkg_out: [float] -optional-
        Inner and outer radius of an annulus to estimate background noise, in 
        pixel unit. Note that P48 pixel scale is 1.012 arcsec per pixel
        The default values are recommended.
        
    Outputs:
    --------
    `targetdir/lightcurves/force_phot_ZTFname_temp.csv: [file]
        force photometry lightcurve [before calibration]
    
    `targetdir/figures/`: [directory]
        cantains visualization of the PSF fitting process; only exist when 
        plot_mod is not None
    '''     
        
    bad_threshold = -500 # Pixels with value < bad_threshold will be counted as bad pixels
    
    ################  read the coordinate and make figure dir #################
    try:
        ra, dec = np.loadtxt(targetdir+'coo.reg')
    except:
        try: 
            ra, dec = np.loadtxt(targetdir+'coo_marshal.reg')
        except:
            print ('Error: no coordinate found!')
            
    imgdir = targetdir+'images_refdiff/'
    psfdir = targetdir+'images_diffpsf/'
    
    try:
        os.stat(targetdir+'figures/')
        figurefiles = glob.glob(targetdir+'figures/*')
        for item in figurefiles:
            os.remove(item)
    except:
        os.mkdir(targetdir+'figures/')
    
    #################### load psf and refdiff image files #####################
    imgfiles = np.array(glob.glob(imgdir+'*.fits'))
    psffiles = np.array(glob.glob(psfdir+'*.fits'))
    arg = np.argsort(imgfiles)
    imgfiles = imgfiles[arg]
    arg = np.argsort(psffiles)
    psffiles = psffiles[arg]
    n = len(imgfiles)
    
    jdobs_ = np.zeros(n)
    filter_ = np.array(['']*n)
    zp_ = np.ones(n)*(99)
    ezp_ = np.ones(n)*(99)
    seeing_ = np.ones(n)*(99)
    programid_ = np.zeros(n)
    field_ = np.zeros(n)
    ccdid_ = np.zeros(n)
    qid_ = np.zeros(n)
    filtercode_ = np.zeros(n)
    
    Fpsf_ = np.ones(n)*(99)
    eFpsf_ = np.ones(n)*(99)
    Fap_ = np.ones(n)*(99)
    rvalue_ = np.zeros(n)
    nbads_ = np.zeros(n)
    nbadbkgs_ = np.zeros(n)
    chi2red_ = np.zeros(n)
    gains_ = np.zeros(n)
    
    ######################## dalta analysis: ight curve ########################
    print ('\n')
    print ('Start fitting forced light curve for %s...'%name)
    for i in range(n):
        if i%20==0:
            print ('In progress: %d in %d...' %(i, n))
        imgpath = imgfiles[i]
        psfpath = psffiles[i]
        pobj = ZTFphot(name, ra, dec, imgpath, psfpath, bad_threshold, r_psf, 
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
        if pobj.status == False:
            continue
        
        pobj.load_source_cutout() 
        if pobj.status == False:
                continue
        pobj.load_bkg_cutout(manual_mask, col_mask_start, col_mask_end,
                             row_mask_start, row_mask_end)
        
        pobj.get_scr_cor_fn()        
        pobj.fit_psf()
        Fpsf_[i] = pobj.Fpsf
        Fap_[i] = pobj.Fap
        eFpsf_[i] = pobj.eFpsf
        rvalue_[i] = pobj.r_value
        nbads_[i] = pobj.nbad
        nbadbkgs_[i] = pobj.nbad_bkg
        chi2red_[i] = pobj.chi2_red
        
        if i%plot_mod==0 or pobj.nbad!=0:
            savepath = targetdir+'/figures/'+repr(i)+'_'+\
                    imgpath.split('/')[-1].split('_sci')[0]
            pobj.plot_cutouts(savepath = savepath)
    print ('\n')
    
    ####################### save the results to a file ########################
    print ('writing light curve to database')
    data = [jdobs_, filter_, seeing_, gains_, zp_, ezp_, Fpsf_, eFpsf_, Fap_, 
            rvalue_, nbads_, nbadbkgs_, chi2red_, programid_, field_,
            ccdid_, qid_, filtercode_]
    
    my_lc = Table(data,names=['jdobs','filter', 'seeing', 'gain', 'zp', 'ezp',
                              'Fpsf', 'eFpsf', 'Fap', 'rvalue', 'nbad',
                              'nbadbkg', 'chi2red', 'programid', 'fieldid',
                              'ccdid', 'qid', 'filterid'])
    
    my_lc.write(targetdir+'/lightcurves/force_phot_'+name+'_output1.fits', overwrite=True)
    
