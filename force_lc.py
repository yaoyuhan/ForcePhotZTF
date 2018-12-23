#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:45:08 2018

@author: yuhanyao
"""
import os
import glob
import numpy as np
import pandas as pd
from copy import deepcopy
from astropy.io import fits
from ztfquery import marshal, query
from ztfquery.io import download_single_url
from phot_class import ZTFphot
from refine_coo import get_refined_coord, get_pos


def download_images_diffpsf_refdiff(targetdir, ra1, dec1, start_jd):
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
        
    ############## Get metadata of all images at this location ################
    zquery = query.ZTFQuery()
    print ('\n')
    print("Querying for metadata...")
    if start_jd == None:
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.01)
    else:
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.01,
                             sql_query='obsjd>'+repr(start_jd))
    out = zquery.metatable
    final_out = out.sort_values(by=['obsjd'])
    final_out.to_csv(targetdir+'/irsafile.csv')
        
    urls, dl_loc = zquery.download_data(nodl=True)
    urls = np.array(urls)
    dl_loc = np.array(dl_loc)
    print ('Trying to download %d images from irsa...' %len(urls)) 
    
    for i in range(len(urls)):
        if i%30 == 0:
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
    print (original_names[~ix_why])
    print ('saving them to missingdata.txt')
    
    with open(targetdir+'/missingdata.txt', 'w') as f:
        for item in original_names[~ix_why]:
            f.write("%s\n" % item)
    f.close()
    
    ###################### check if files can be opened #######################
    print ('checking if all files can be opened...')
    imgdir = targetdir+'/images_refdiff/'
    psfdir = targetdir+'/images_diffpsf/'
    imgfiles = np.array(glob.glob(imgdir+'*.fits'))
    arg = np.argsort(imgfiles)
    imgfiles = imgfiles[arg]
    psffiles = np.array(glob.glob(psfdir+'*.fits'))
    arg = np.argsort(psffiles)
    psffiles = psffiles[arg]
    assert len(imgfiles)==len(psffiles)
    n = len(imgfiles)
        
    for i in range(n):
        if i%30 == 0:
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
    

def get_force_lightcurve(name, targetdir = 'default',
                         before_marshal_detection = None,
                         detection_jd = None,
                         plot_mod = None,
                         bad_threshold = -500,
                         recenter_coo = True,
                         ndays_before_peak = 8, ndays_after_peak = 10,
                         manual_mask = False, 
                         col_mask_start = 0, col_mask_end = 0,                  
                         row_mask_start = 0, row_mask_end = 0,
                         bkg_r_in = 25, bkg_r_out = 30):
    '''
    Get force psf photometry light curve for ZTF a source
    
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
         
    plot_mod: [int | None] -optional-
        Visualization of the psf fitting every plot_mod images
        All figures will be saved to `targetdir/figures/`
        - `plot_mod=None`: do not visualize the fitting
        - `plot_mod=1`: visualize every image 
        
    bad_thre: [int, <0] -optional-
        Pixels with value < bad_threshold will be counted as bad pixels
        
    recenter_coo: [bool] -optional-
        Re-determine the coordinate of the target by taking the median of all
        target positions within 
        [jd_peak - ndays_before_peak, jd_peak + ndays_after_peak]
        - `recenter_coo=False`: use the coordinate on Marshal webpage
    
    ndays_before_peak, ndays_after_peak: [int] -optional-
        see recenter_coo; Only useful when recenter_coo is True
        
    manual_mask: [bool] -optional-
        manually mask some pixels within the 25*25 cutout at every epoch. 
        If true, masked region will be a square of
        [col_mask_start:col_mask_end, row_mask_start:row_mask_end]
        currently only support square mask.
        Set manual_mask = True when there is bad subtraction at the same region
        on every image (e.g., nucleus of the host galaxy)
    
    col_mask_start, col_mask_end, row_mask_start, row_mask_end: [int, >=0, <25] -optional-
        see manual_mask; Only useful when manual_mask = True
        
    bkg_r_in, bkg_r_out: [float] -optional-
        Inner and outer radius of an annulus to estimate background noise, in 
        pixel unit. Note that the pixel scale of ZTF Camera is 1.012 arcsec per pixel.
        The default values are recommended.
        
    Returns:
    -------
    targetdir: string
        the name of the directory of this target
    
    Outputs:
    --------
    `targetdir/lightcurves`: [directory]
        contains both marshal lightcurve and force photometry lightcurve
        
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
        
    `targetdir/figures/`: [directory]
        cantains visualization of the PSF fitting process; only exist when 
        plot_mod is not None
    '''
         
    name = str(name)
    
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
            
    ####################### download marshal lightcurve #######################
    try:
        os.stat(targetdir+'lightcurves/')
    except:
        os.mkdir(targetdir+'lightcurves/')
    marshal.download_lightcurve(name, dirout = targetdir+'lightcurves/')
    
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
        if detection_jd is not None:
            start_jd = detection_jd - before_marshal_detection
        else:
            marshal_lc = pd.read_csv(targetdir+'lightcurves/'+ \
                                     'marshal_lightcurve_'+name+'.csv')    
            detection_ix = np.where(marshal_lc['magpsf'].values!=99)[0][0]
            detection_jd = marshal_lc['jdobs'][detection_ix]
            start_jd = detection_jd - before_marshal_detection
    download_images_diffpsf_refdiff(targetdir, ra1, dec1, start_jd)
        
    ################# re-center the coordinate of the source ##################
    if recenter_coo == False:
        ra = deepcopy(ra1)
        dec = deepcopy(dec1)
    else:
        print ('Determining the coordinate based on observations around peak...')
        marshal_lc = pd.read_csv(targetdir+'lightcurves/'+ \
                                 'marshal_lightcurve_'+name+'.csv')  
        mags = marshal_lc['magpsf'].values
        peak_ix = np.where(mags == mags.min())[0][0]
        peak_jd = marshal_lc['jdobs'][peak_ix]
        fade_ix = np.where(marshal_lc['magpsf'].values!=99)[0][-1]
        fade_jd = marshal_lc['jdobs'][fade_ix]
        if detection_jd == None:
            detection_ix = np.where(marshal_lc['magpsf'].values!=99)[0][0]
            detection_jd = marshal_lc['jdobs'][detection_ix]
            
        if (peak_jd - detection_jd) < ndays_before_peak:
            ndays_before_peak = peak_jd - detection_jd
            print ('setting ndays_before_peak to %.2f'%ndays_before_peak)
        if (fade_jd - peak_jd) < ndays_after_peak:
            ndays_after_peak = fade_jd - peak_jd
            print ('setting ndays_after_peak to %.2f' %ndays_after_peak)
    
        ra, dec = get_refined_coord(name, ra1, dec1, bad_threshold, targetdir, peak_jd,
                                    ndays_before_peak, ndays_after_peak)
        
    ################### start getting the PSF photometry ! ####################
    imgdir = targetdir+'/images_refdiff/'
    psfdir = targetdir+'/images_diffpsf/'
    
    if plot_mod != None:
        try:
            os.stat(targetdir+'/figures/')
            figurefiles = glob.glob(targetdir+'/figures/*')
            for item in figurefiles:
                os.remove(item)
        except:
            os.mkdir(targetdir+'/figures/')
    
    # load psf and refdiff image files
    imgfiles = np.array(glob.glob(imgdir+'*.fits'))
    psffiles = np.array(glob.glob(psfdir+'*.fits'))
    arg = np.argsort(imgfiles)
    imgfiles = imgfiles[arg]
    arg = np.argsort(psffiles)
    psffiles = psffiles[arg]
    n = len(imgfiles)
    
    jdobs_ = np.zeros(n)
    filter_ = np.array(['']*n)
    Fpsf_ = np.ones(n)*(99)
    eFpsf_ = np.ones(n)*(99)
    Fpsf_over_F0_ = np.ones(n)*(99)
    eFpsf_over_F0_ = np.ones(n)*(99)
    magpsf_ = np.ones(n)*(99)
    emagpsf_ = np.ones(n)*(99)
    rvalue_ = np.zeros(n)
    nbads_ = np.zeros(n)
    r_thresholds = np.array([0.3, 0.35, 0.4])
    limmags_ = np.ones((n, len(r_thresholds)))*(99)
    
    # data analysis: ight curve
    print ('\n')
    print ('Start extracting light curve for %s...'%name)
    for i in range(n):
        if i%20==0:
            print ('In progress: %d in %d...' %(i, n))
        imgpath = imgfiles[i]
        psfpath = psffiles[i]
        pobj = ZTFphot(name, ra, dec, imgpath, psfpath, bad_threshold)
        if pobj.status == False:
            continue
        pobj.load_source_cutout() 
        pobj.load_bkg_cutout(bkg_r_in, bkg_r_out)
        pobj.get_scr_cor_fn(manual_mask, col_mask_start, col_mask_end,
                            row_mask_start, row_mask_end)        
        
        pobj.fit_psf()
        Fpsf_[i] = pobj.Fpsf
        eFpsf_[i] = pobj.e_Fpsf
        Fpsf_over_F0_[i] = pobj.Fpsf_over_flux0
        eFpsf_over_F0_[i] = pobj.e_Fpsf_over_flux0
        magpsf_[i] = pobj.magpsf
        emagpsf_[i] = pobj.emagpsf
        jdobs_[i] = pobj.obsjd
        filter_[i] = pobj.filter
        nbads_[i] = pobj.nbad
        rvalue_[i] = pobj.r_value
        
        pobj.find_upper_limit(r_thresholds = r_thresholds)
        limmags_[i] = pobj.limmags
        
        if plot_mod!=None:
            if i%plot_mod==0:
                savepath = targetdir+'/figures/'+repr(i)+'_'+\
                            imgpath.split('/')[-1].split('_sci')[0]
                pobj.plot_cutouts(savepath = savepath)
    print ('\n')
    
    ####################### save the results to a file ########################
    print ('wring light curve to database')
    data = np.vstack([jdobs_, filter_, 
                      Fpsf_, eFpsf_, 
                      Fpsf_over_F0_, eFpsf_over_F0_, 
                      magpsf_, emagpsf_, 
                      rvalue_, limmags_.T, 
                      nbads_]).T
    noid = jdobs_==0
    data = data[~noid]
    my_lc = pd.DataFrame(data,columns=['jdobs','filter', 'Fpsf', 'e_Fpsf', 
                                       'Fpsf/F0','e_Fpsf/F0', 
                                       'magpsf', 'e_magpsf', 'r_value', 
                                       'limmag_r=0.3', 'limmag_r=0.35',
                                       'limmag_r=0.4', 'nbad'])
    my_lc.to_csv(targetdir+'/lightcurves/force_phot_'+name+'.csv',
                 index = False)
   
    return targetdir
        
        
