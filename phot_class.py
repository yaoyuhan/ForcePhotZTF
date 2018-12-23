#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:36:23 2018

@author: Yuhan Yao
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from copy import deepcopy
from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAnnulus
from image_registration import chi2_shift_iterzoom#, chi2_shift

import random
random.seed(0)


def mylinear_fit(x, y, yerr, npar=2):
    '''
    Ref: 
        1. Numerical Recipes, 3rd Edition, p745, 781 - 782
        2. http://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_deliverables.pdf, p37
    '''
    assert len(x) == len(y)
    assert len(y) == len(yerr)
    
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxy = np.sum(x * y)
    Sxx = np.sum(x**2)
    N = len(x)
    
    Sx_sigma = np.sum(x * yerr**2)
    Sxx_sigma = np.sum(x**2 * yerr**2)
    S_sigma = np.sum(yerr**2)
    
    if npar==1:
        Fpsf = Sxy / Sxx
        e_Fpsf = np.sqrt(Sxx_sigma) / Sxx
        a = 0
    elif npar==2:
        Fpsf = (N * Sxy - Sx * Sy) / (N * Sxx - Sx**2)
        a = (Sxx * Sy - Sx * Sxy) / (N * Sxx - Sx**2)
        e_Fpsf = np.sqrt(N**2*Sxx_sigma - 2*N*Sx*Sx_sigma + Sx**2*S_sigma) / (N * Sxx - Sx**2)
        
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    pearson_r = np.sum( (x - x_mean) * (y - y_mean) ) / np.sqrt(np.sum( (x - x_mean)**2 )) / np.sqrt(np.sum( (y - y_mean)**2 ))
    return Fpsf, e_Fpsf, a, pearson_r
    

def my_sigma_clip_with_mask(residual_fn, ok_ix, sigma = 3):
    data = residual_fn[ok_ix]
    filtered_data = sigma_clip(data, sigma = sigma)
    ok_ravel = ok_ix.ravel()
    newarray = np.linspace(0, 624, num=625, dtype=int)
    newmask_loc = newarray[ok_ravel][filtered_data.mask]
    sigma_mask_ravel = np.zeros(625, dtype = bool)
    sigma_mask_ravel[newmask_loc] = True
    sigma_mask = sigma_mask_ravel.reshape(25, 25)
    return sigma_mask
    

class ZTFphot(object):
    
    def __init__(self, name, ra, dec, imgpath, psfpath, bad_thre):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.imgpath = imgpath
        self.psfpath = psfpath  
        self.bad_thre = bad_thre
    
        hd = fits.open(imgpath)[1].header
        dt = fits.open(imgpath)[1].data
        n_dty = dt.shape[0]
        n_dtx = dt.shape[1]
        w = wcs.WCS(hd)
        world =  np.array([[ra, dec]], np.float_)
        pixcrd = w.wcs_world2pix(world, 1)
        # need to subtract 1 !
        # since physical coordinate in python starts from 0, not 1
        pixX = pixcrd[0, 0]-1 
        pixY = pixcrd[0, 1]-1

        pixXint = int(np.rint(pixX))
        pixYint = int(np.rint(pixY))
        
        if pixXint<0 or pixYint<0 or pixYint>n_dty or pixXint>n_dtx:
            self.status = False
            print ('Set status to False -- Target outside of image!')
        elif pixXint<12 or pixYint<12 or pixYint>(n_dty-12) or pixXint>(n_dtx-12):
            print ('Set status to False -- Target on the edge of the image!')
            self.status = False
        elif dt[pixYint, pixXint]<bad_thre:
            self.status = False
            print ('Set status to False -- Bad pixel in the center of PSF!')
        else:
            self.status = True
        
        zp = hd['MAGZP']
        e_zp = hd['MAGZPUNC']
        obsjd = hd['OBSJD']
        flux0 = 10**(zp/2.5)
        e_flux0 = flux0  / 2.5 * np.log(10) * e_zp
        
        self.pixX = pixX
        self.pixY = pixY
        self.pixXint = pixXint
        self.pixYint = pixYint
        self.obsjd = obsjd
        self.zp = zp
        self.e_zp = e_zp
        self.flux0 = flux0
        self.e_flux0 = e_flux0
        self.filter = hd['FILTER'][4]
        self.gain = hd['GAIN']
            
        # load psf cutout
        psf_fn = fits.open(psfpath)[0].data 
        self.psf_fn = psf_fn  
        
        
    def load_source_cutout(self):
        '''
        imgpath = pobj.imgpath
        pixX = pobj.pixX
        pixY = pobj.pixY        
        pixXint = pobj.pixXint
        pixYint = pobj.pixYint
        bad_thre = pobj.bad_thre
        '''
        imgpath = self.imgpath        
        pixX = self.pixX
        pixY = self.pixY        
        pixXint = self.pixXint
        pixYint = self.pixYint
        bad_thre = self.bad_thre
        
        dt = fits.open(imgpath)[1].data        
        scr_fn_1 = dt[pixYint - 13 : pixYint + 14, 
                      pixXint - 13 : pixXint + 14]
        xoff_tobe = pixX - pixXint
        yoff_tobe = pixY - pixYint
        scr_fn_ = ndimage.shift(scr_fn_1, [-yoff_tobe, -xoff_tobe], order=3, 
                                mode='reflect', cval=0.0, prefilter=True)
        scr_fn = scr_fn_[1:-1, 1:-1]      
        
        bad_mask = scr_fn <= bad_thre
        nbad = np.sum(bad_mask)
        scr_fn[bad_mask] = np.nan
        self.bad_mask = bad_mask
        self.nbad = nbad
        self.scr_fn = scr_fn
        
        if nbad!=0:
            print ('%d bad pixels in 25*25 source frame' %nbad)
        
        
    def find_optimal_coo(self):
        psf_fn = self.psf_fn
        scr_fn = self.scr_fn
        pixX = self.pixX
        pixY = self.pixY
        imgpath = self.imgpath
        
        hd = fits.open(imgpath)[1].header
        w = wcs.WCS(hd)
        xoff, yoff, exoff, eyoff = chi2_shift_iterzoom(psf_fn, scr_fn)
        pixX_cor = pixX + xoff 
        pixY_cor = pixY + yoff 
        pixel =  np.array([[pixX_cor+1, pixY_cor+1]], np.float_)
        newcrd = w.wcs_pix2world(pixel, 1)
        ra_cor = newcrd[0][0] 
        dec_cor = newcrd[0][1]
        self.ra_cor = ra_cor
        self.dec_cor = dec_cor
        
        
    def load_bkg_cutout(self, bkg_r_in, bkg_r_out):
        '''
        imgpath = pobj.imgpath
        pixX = pobj.pixX
        pixY = pobj.pixY  
        bad_thre = pobj.bad_thre
        '''
        imgpath = self.imgpath              
        pixX = self.pixX
        pixY = self.pixY
        bad_thre = self.bad_thre
        
        dt = fits.open(imgpath)[1].data        
        positions = [(pixX, pixY)]
        annulus = CircularAnnulus(positions, r_in = bkg_r_in, r_out = bkg_r_out)
        annulus_masks = annulus.to_mask(method='center')
        annulus_data = annulus_masks[0].multiply(dt)
        
        bkg_fn = deepcopy(annulus_data)
        bad_bkg_mask = annulus_data <= bad_thre
        bkg_fn[bad_bkg_mask] = np.nan
        nbad_bkg = np.sum(bad_bkg_mask)
            
        self.bad_bkg_mask = bad_bkg_mask
        self.nbad_bkg = nbad_bkg
        
        setnan = annulus_masks[0].data==0
        bkg_fn[setnan] = np.nan
        
        bkgstd = np.nanstd(bkg_fn) 
        
        self.bkgstd = bkgstd
        self.bkg_fn = bkg_fn
        
        
    def get_scr_cor_fn(self, manual_mask, col_mask_start, col_mask_end,
                       row_mask_start, row_mask_end):
        bad_mask = self.bad_mask
        scr_fn = self.scr_fn
        
        if manual_mask == True:     
            manual_mask = np.zeros((25, 25), dtype = bool)
            manual_mask[col_mask_start:col_mask_end, row_mask_start:row_mask_end] = True
            bad_mask[manual_mask] = True
        self.bad_mask = bad_mask
            
        scr_cor_fn = deepcopy(scr_fn)
        scr_cor_fn[bad_mask] = np.nan 
        
        self.scr_cor_fn = scr_cor_fn
    
        
    def fit_psf(self):
        '''
        psf_fn = pobj.psf_fn
        scr_cor_fn = pobj.scr_cor_fn
        bad_mask = pobj.bad_mask
        bkgstd = pobj.bkgstd
        zp = pobj.zp
        e_zp = pobj.e_zp
        gain  = pobj.gain
        '''
        psf_fn = self.psf_fn  
        scr_cor_fn = self.scr_cor_fn            
        bad_mask = self.bad_mask
        bkgstd = self.bkgstd
        flux0 = self.flux0
        e_flux0 = self.e_flux0
        gain = self.gain
                
        psf_ravel = psf_fn.ravel()
        _psf_ravel = psf_fn[~bad_mask]
        _scr_cor_ravel = scr_cor_fn[~bad_mask]
        _yerrsq = _scr_cor_ravel / gain + bkgstd**2 
        _yerr = np.sqrt(_yerrsq)

        # two-parameter fit
        Fpsf, e_Fpsf, apsf, pearson_r = mylinear_fit(_psf_ravel, _scr_cor_ravel, _yerr, npar=2)
    
        model_ravel = psf_ravel * Fpsf + apsf
        model_fn = np.reshape(model_ravel, (25,25))
        model_fn[bad_mask] = np.nan
        _model_ravel = model_fn[~bad_mask]
        
        Fpsf_over_flux0 = Fpsf/flux0
        e_Fpsf_over_flux0 = np.sqrt((e_Fpsf/ flux0)**2 + (Fpsf/flux0**2 * e_flux0)**2)
    
        magpsf = 99
        emagpsf = 99
        if Fpsf > 0:
            magpsf = round(-2.5*np.log10(Fpsf/flux0), 5)
            emagpsf = round(2.5 / np.log(10) * e_Fpsf_over_flux0 / Fpsf_over_flux0, 5)
                            
        self._psf_ravel = _psf_ravel
        self._scr_cor_ravel = _scr_cor_ravel
        self.Fpsf = Fpsf
        self.e_Fpsf = e_Fpsf
        self.apsf = apsf
        self.model_fn = model_fn
        self._model_ravel = _model_ravel
        self.r_value = round(pearson_r, 3)   
        self.magpsf = magpsf
        self.emagpsf = emagpsf
        self.Fpsf_over_flux0 = Fpsf_over_flux0
        self.e_Fpsf_over_flux0 = e_Fpsf_over_flux0
            
        
    def _find_upper_lim_individual(self, r_thre):
        '''
        zp = pobj.zp
        _psf_ravel = pobj._psf_ravel
        _model_ravel = pobj._model_ravel
        _scr_cor_ravel = pobj._scr_cor_ravel
        '''
        zp = self.zp
        _psf_ravel = self._psf_ravel
        _model_ravel = self._model_ravel
        _scr_cor_ravel = self._scr_cor_ravel
        _noise_ravel = _scr_cor_ravel - _model_ravel
        num =  len(_psf_ravel)
        
        m = 16
        m_step = 1
        absr = 1
        while absr > r_thre:
            if m==21:
                # print ("limmag investigation break at m=21")
                break   
            m += m_step
            flux = 10**(0.4*(zp - m))
            _sigmal_ravel = _psf_ravel * flux    
            _obs_ravel = _sigmal_ravel + _noise_ravel
            b, eb, a, r_value = mylinear_fit(_psf_ravel, _obs_ravel, np.ones(num), npar=2)
            absr = r_value      

        if m==21 and absr > r_thre:
            return m
        
        else:
            m -= 1.1
            m_step = 0.1
            absr = 1
            while absr > r_thre:
                m += m_step
                flux = 10**(0.4*(zp - m))
                _sigmal_ravel = _psf_ravel * flux    
                _obs_ravel = _sigmal_ravel + _noise_ravel
                b, eb, a, r_value = mylinear_fit(_psf_ravel, _obs_ravel, np.ones(num), npar=2)
                absr = r_value      
          
            m -= 0.11
            m_step = 0.01
            absr = 1
            while absr > r_thre:
                m += m_step
                flux = 10**(0.4*(zp - m))
                _sigmal_ravel = _psf_ravel * flux    
                _obs_ravel = _sigmal_ravel + _noise_ravel
                b, eb, a, r_value = mylinear_fit(_psf_ravel, _obs_ravel, np.ones(num), npar=2)
                absr = r_value 
            
            limmag = m - 0.01
            return limmag
        
        
    def find_upper_limit(self, r_thresholds):
        limmags = np.ones(len(r_thresholds))
        for i in range(len(r_thresholds)):
            r_thre = r_thresholds[i]
            limmags[i] = self._find_upper_lim_individual(r_thre)
        self.limmags = limmags
    
    
    def plot_cutouts(self, savepath=None, cmap_name = 'viridis'):
        scr_fn = self.scr_fn
        _scr_cor_ravel = self._scr_cor_ravel
        scr_cor_fn = self.scr_cor_fn
        model_fn = self.model_fn
        _model_ravel = self._model_ravel
        e_flux = self.e_Fpsf
        bad_mask = self.bad_mask
        flux = self.Fpsf
        filtername = self.filter
        
        fig, ax = plt.subplots(3, 4, figsize=(14, 10.5))
        matplotlib.rcParams.update({'font.size': 15})
        norm = ImageNormalize(stretch=SqrtStretch())
        if np.sum(bad_mask) != 0:
            ax[0,0].imshow(scr_fn, cmap = cmap_name, origin='lower', norm=norm)
            ax[0,0].set_title('Unmasked: '+filtername, fontsize=15)
        else:
            ax[0,0].set_axis_off()
        norm2 = ImageNormalize(stretch=SqrtStretch())
        ax[0,1].imshow(scr_cor_fn, cmap = cmap_name, origin='lower', norm=norm2)
        ax[0,2].imshow(model_fn, cmap = cmap_name, origin='lower', norm=norm2)
        ax[0,1].set_title('Data', fontsize=15)
        ax[0,2].set_title('PSF model', fontsize=15)
        
        norm1 = ImageNormalize(stretch=SqrtStretch())
        ax[0,3].imshow(scr_cor_fn-model_fn, cmap = cmap_name, origin='lower', norm=norm1)
        ax[0,3].set_title('Residual', fontsize=15)
        plt.tight_layout()
        
        ax4 = plt.subplot2grid((9, 1), (3, 0), rowspan=4)
        ax4.plot(_scr_cor_ravel, 'r--', label='obs')
        ax4.plot(_model_ravel, 'b--', label='fitted')
        ax4.set_xlim(0-5, 625+5)
        ax4.legend(loc='upper right')
        ylims = ax4.get_ylim()
        ax4.set_xticklabels([])
        yloc = ylims[0] + (ylims[1] - ylims[0])*0.7
        plt.text(0 ,yloc, 'flux = %.1f, e_flux = %.1f'%(flux, e_flux))
        
        ax5 = plt.subplot2grid((9, 1), (7, 0), rowspan=2)
        ax5.plot(_scr_cor_ravel - _model_ravel, '.k', label='obs - fitted')
        ax5.set_xlim(0-5, 625+5)
        ax5.legend(loc='upper right')
        ylims = ax5.get_ylim()
        ymaxabs = max(abs(ylims[0]), abs(ylims[1]))
        ax5.set_ylim(-1*ymaxabs, ymaxabs)
        plt.plot([0, 625], [0,0], color='grey', linewidth = 2, alpha= 0.5)
    
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()