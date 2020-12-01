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

import astropy.io.ascii as asci
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from ztfquery import query
from ztfquery import marshal
from ztfquery.io import download_single_url

from penquins import Kowalski

from ForcePhotZTF.keypairs import get_keypairs
from ForcePhotZTF.phot_class import ZTFphot

DEFAULT_AUTHs = get_keypairs()
DEFAULT_AUTH_marshal = DEFAULT_AUTHs[0]
DEFAULT_AUTH_kowalski = DEFAULT_AUTHs[1]


def download_images_diffpsf_refdiff(targetdir, ra1, dec1, start_jd = None, 
                                    end_jd = None, open_check = True):
    '''
    Download subtracted images and psf images from irsa

    This function will create two folders to store fits files downloaded from ipac:
	difference images go to --> targetdir/images_refdiff/
	psf images go to --> targetdir/images_diffpsf/

    ra1, dec1 are the coordinates of the target

    set start_jd and end_jd = None if you want all images in ZTF history
        otherwise, only download images between (start_jd, end_jd)

    set open_check = True, the function will try to opeb all files in the final step.
    	Sometimes (although very seldom) the fits file can be broken.
        
    Outputs:
    --------
    `targetdir/images_refdiff/`: [directory]
        contains difference images on which PSF photometry is performed
        
    `targetdir/images_diffpsf/`: [directory]
        contains images with psf models on each epoch
        
    `targetdir/irsafile.csv` & `targetdir/missingdata.txt`: [file]
        by-products in the process of downloading from irsa
    '''
    try:
        os.stat(targetdir)
    except:
        os.mkdir(targetdir) 
	
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
    if start_jd == None and end_jd==None:
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.003)
    else:
        if start_jd!=None and end_jd==None:
            sql_query='obsjd>'+repr(start_jd)
        elif start_jd==None and end_jd!=None:
            sql_query='obsjd<'+repr(end_jd)
        elif start_jd!=None and end_jd!=None:
            sql_query='obsjd<'+repr(end_jd)+'+AND+'+'obsjd>'+repr(start_jd)
        zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.003,
                             sql_query=sql_query)
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
    
    Outputs:
    --------
    `targetdir/lightcurves/marshal_lightcurve_ZTFname.csv`: [file]
        marshal lightcurve csv file
        
    `targetdir/marshal_lc_plotZTFname.png`: [file]
        marshal lightcurve plot
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
        mtb = tables[len(tables)-1]
        mtb = mtb.drop([0], axis=1)
        mtb.to_csv(targetdir+'lightcurves'+'/marshal_lc_'+name+'.csv',header=False, index = False)
    
    ###################### visualize marshal lightcurve #######################
    mtb = asci.read(targetdir+'lightcurves'+'/marshal_lc_'+name+'.csv')
    if 'absmag' in mtb.colnames:
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
    ax2.set_xlabel('jd')
    ax2.set_ylabel('mag')
    ax1.set_ylabel('mag')
    plt.tight_layout()
    plt.savefig(targetdir+'marshal_lc_plot'+name+'.png')
    plt.close()   
    

def get_pos(name):
    """
    Get position of target
    """
    m = marshal.MarshalAccess()
    m.load_target_sources()
    coords = m.get_target_coordinates(name)
    ra = coords.ra.values[0]
    dec = coords.dec.values[0]
    return ra, dec


def astrometry_spread(name, targetdir):
    ra1, dec1 = np.loadtxt(targetdir+'coo_marshal.reg') 
    ra2, dec2 = np.loadtxt(targetdir+'coo_kowalski.reg') 
    
    tb_alerts = pd.read_csv(targetdir+'alert_coo.csv')
    
    radius = 0.1
    fontsize = 12
    c1 = SkyCoord(ra = ra1, dec = dec1, unit = 'degree')
    c2 = SkyCoord(ra = ra2, dec = dec2, unit = 'degree')
    coords = SkyCoord(ra = tb_alerts['ra'].values, dec = tb_alerts['dec'].values, unit='degree')
    
    xs = []
    ys = []
    for i in range(len(coords)):
        sep1 = c2.separation(coords[i]).arcsec
        pos1 = c2.position_angle(coords[i]).rad
        x1 = sep1 * np.cos(pos1)
        y1 = sep1 * np.sin(pos1)
        xs.append(x1)
        ys.append(y1)
    xs = np.array(xs)
    ys = np.array(ys)
    
    sep2 = c2.separation(c1).arcsec
    pos2 = c2.position_angle(c1).rad
    x2 = sep2 * np.cos(pos2)
    y2 = sep2 * np.sin(pos2)
    
    plt.figure(figsize = (6, 6))
    for i in range(len(xs)):
        plt.plot([0, xs[i]], [0, ys[i]], 'k-', alpha=0.1, linewidth = 2.5, zorder=1)
        plt.plot(xs[i], ys[i], 'k.', zorder=1)
    plt.plot(0, 0, 'k.', label = '%d alert detections'%len(tb_alerts))
    plt.plot(0, 0, 'r.', label='re-centered position', zorder=2)
    plt.xlabel(r'$\Delta$'+'RA (arcsec)')
    plt.ylabel(r'$\Delta$'+'DEC (arcsec)')
    plt.title(name)
    
    plt.plot([0,x2], [0, y2], 'm-', zorder=2)
    plt.plot(x2, y2, 'm.', label='marshal position', zorder=2)
    
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
    plt.savefig(targetdir+'astrometry.png')
    plt.close()


def recenter_pos_kowalski(targetdir, name):
    k = Kowalski(username=DEFAULT_AUTH_kowalski[0], password=DEFAULT_AUTH_kowalski[1], verbose=False)
    qu = {"query_type": "general_search", 
          "query": "db['ZTF_alerts'].find({'objectId': {'$eq': '%s'}})"%name}
        
    r = k.query(query=qu)
    
    if 'result_data' in r.keys():
        rdata = r['result_data']
        rrdata = rdata['query_result']
        n = len(rrdata)
        ras = np.zeros(n)
        decs = np.zeros(n)
        for i in range(n):
            rrrcoo = rrdata[i]['coordinates']['radec_deg']
            ras[i] = rrrcoo[0]
            decs[i] = rrrcoo[1]
            
        tb = Table([ras, decs], names = ['ra', 'dec'])
        tb.write(targetdir+'alert_coo.csv', overwrite=True)
        
        ra = np.median(ras)
        dec = np.median(decs)
        
        
        return ra, dec
    else:
        print ('Kowalski query is not succesful for %s'%name)


def get_coo_ZTFtarget(name, targetdir):
    '''
    Preparation for force psf photometry light curve for ZTF a source:
    
    Parameters:
    -----------
    name: [str]
        Name of a target on the marshal.
        
    targetdir: [str] 
        Directory where the data should be stored. 
                                  
    recenter_coo: [bool] -optional-
        Re-determine the coordinate of the target by taking the median of all
        target positions in alert packet
         
    Returns:
    -------
    None
    
    Outputs:
    --------
    `targetdir/coo_marshal.reg`: [file]
        marshal coordinate of the source
    
    `targetdir/coo_kowalksi.reg`: [file]
        re-centered coordinate; only exist when recenter_coo == True
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
    
    ################ get the marshal coordinate of this target ################
    try:
        ra1, dec1 = np.loadtxt(targetdir+'/coo_marshal.reg')
    except:
        print ("Start getting marshal coordinate for %s" %name)
        ra1, dec1 = get_pos(name)
        print ('%s: ra=%f, dec=%f'%(name, ra1, dec1))
        c1 = SkyCoord(ra = ra1, dec = dec1, unit = 'degree')
        print (c1.ra.to_string(u.hour) + ' ' + c1.dec.to_string(u.degree))
        np.savetxt(targetdir+'/coo_marshal.reg', [ra1, dec1])
        
    ################# re-center the coordinate of the source ##################
    try:
        ra2, dec2 = np.loadtxt(targetdir+'/coo_kowalski.reg')
    except:
        print ('\n')
        print ("Recentering coordinate for %s from Kowalki" %name)
        ra2, dec2 = recenter_pos_kowalski(targetdir, name)
        print ('%s: ra=%f, dec=%f'%(name, ra2, dec2))
        c2 = SkyCoord(ra = ra2, dec = dec2, unit = 'degree')
        print (c2.ra.to_string(u.hour) + ' ' + c2.dec.to_string(u.degree))
        np.savetxt(targetdir+'/coo_kowalski.reg', [ra2, dec2])
        
        
def get_cutout_data(name, targetdir, ra, dec,
                    r_psf = 3, r_bkg_in = 25, r_bkg_out = 30, verbose = False):
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
    
    path_cutouts = []
    psf_cutouts = []
    img_cutouts = []
    eimg_cutouts = []
        
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
        
        nbads_[i] = pobj.nbad
        nbadbkgs_[i] = pobj.nbad_bkg
        bkgstd_[i] = pobj.bkgstd
        bkgmed_[i] = pobj.bkgmed
        
        path_cutouts.append(imgpath.split('/')[-1])
        psf_cutouts.append(pobj.psf_fn[~pobj.bad_mask])
        img_cutouts.append(pobj.scr_cor_fn[~pobj.bad_mask])
        eimg_cutouts.append(pobj.yerrs)
    print ('\n')
    
    ####################### save the results to a file ########################
    diffimgname = np.array([x.split('/')[-1]for x in imgfiles])
    psfimgname = np.array([x.split('/')[-1]for x in psffiles])
    
    data = [jdobs_, filter_, seeing_, gains_, zp_, ezp_, 
            programid_, field_, ccdid_, qid_, filtercode_, 
            moonra_, moondec_, moonillf_, moonphas_, airmass_,
            nbads_, nbadbkgs_, bkgstd_,  bkgmed_, diffimgname, psfimgname]
    
    my_lc = Table(data,names=['jdobs','filter', 'seeing', 'gain', 'zp', 'ezp',
                              'programid', 'fieldid', 'ccdid', 'qid', 'filterid', 
                              'moonra', 'moondec', 'moonillf', 'moonphase', 'airmass',
                              'nbad', 'nbadbkg', 'bkgstd', 'bkgmed', 'diffimgname', 
                              'psfimgname'])
    
    mylc = my_lc[status_==1]
    print ('writing data to database')
    mylc.write(targetdir+'/lightcurves/force_phot_'+name+'_info.fits', overwrite=True)
    
    assert len(path_cutouts) == len(mylc)
    path_cutouts = np.array(path_cutouts)
    psf_cutouts = np.array(psf_cutouts)
    img_cutouts = np.array(img_cutouts)
    xs = []
    ys = []
    eys = []
    ps = []
    index = []
    for i in range(len(path_cutouts)):
        for x in psf_cutouts[i]:
            xs.append(x)
            ps.append(path_cutouts[i])
        for y in img_cutouts[i]:
            ys.append(y)
            index.append(i)  
        for ey in eimg_cutouts[i]:
            eys.append(ey)
        
    my_dt = Table([index, xs, ys, eys, ps], names = ['index', 'x', 'y', 'ey', 'path'])
    my_dt.write(targetdir+'/lightcurves/xydata_'+name+'.fits', overwrite=True)  