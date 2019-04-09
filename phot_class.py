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
from scipy.interpolate import Rbf
from copy import deepcopy

from astropy import wcs
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from photutils import CircularAnnulus
from image_registration import chi2_shift_iterzoom#, chi2_shift

# ref: https://github.com/Caltech-IPAC/ztf/blob/master/src/pl/perl/forcedphotometry.pl


def mylinear_fit(x, y, yerr, npar = 2):
    '''
    Ref: 
        1. Numerical Recipes, 3rd Edition, p745, 781 - 782
        2. http://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_deliverables.pdf, p38
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
    # x_mean = np.mean(x)
    # y_mean = np.mean(y)
    # pearson_r = np.sum( (x - x_mean) * (y - y_mean) ) / np.sqrt(np.sum( (x - x_mean)**2 )) / np.sqrt(np.sum( (y - y_mean)**2 ))
    return Fpsf, e_Fpsf, a


class ZTFphot(object):
    
    def __init__(self, name, ra, dec, imgpath, psfpath,
                 r_psf=12, r_bkg_in=5, r_bkg_out=15, verbose=False):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.imgpath = imgpath
        self.psfpath = psfpath  
        self.r_psf = r_psf
        self.r_bkg_in = r_bkg_in
        self.r_bkg_out = r_bkg_out
        self.bad_threshold = -500
        self.length = 2*r_psf + 1
        self.verbose = verbose
        self.stampupsamplefac = 5
        self.newlength = self.stampupsamplefac * self.length
    
        hd = fits.open(imgpath)[1].header
        dt = fits.open(imgpath)[1].data
        n_dty = dt.shape[0]
        n_dtx = dt.shape[1]
        w = wcs.WCS(hd)
        world =  np.array([[ra, dec]], np.float_)
        pixcrd = w.wcs_world2pix(world, 0)
        pixX = pixcrd[0, 0]
        pixY = pixcrd[0, 1]
        
        self.pixX = pixX
        self.pixY = pixY
        self.n_dty = n_dty
        self.n_dtx = n_dtx
        
        
        if np.isnan(pixX)==1 or np.isnan(pixY)==1:
            self.status = False
            if self.verbose==True:
                print ('Set status to False -- Target outside of image! %s'%(imgpath.split('/')[-1]))

        else:
            pixXint = int(np.rint(pixX))
            pixYint = int(np.rint(pixY))
            self.pixXint = pixXint
            self.pixYint = pixYint
            
            # require no bad pixels in the central 3*3 small cutout
            small_cutout = dt[pixYint-1: pixYint+2, pixXint-1: pixXint+2]
        
            if pixXint<0 or pixYint<0 or pixYint>n_dty or pixXint>n_dtx:
                self.status = False
                if self.verbose==True:
                    print ('Set status to False -- Target outside of image! %s'%(imgpath.split('/')[-1]))
            elif pixXint < r_psf or pixYint < r_psf or pixYint >= (n_dty - r_psf) or pixXint >= (n_dtx - r_psf):
                self.status = False
                if self.verbose==True:
                    print ('Set status to False -- Target on the edge of the image! %s'%(imgpath.split('/')[-1]))
            elif np.sum(small_cutout < self.bad_threshold) != 0:
                self.status = False
                if self.verbose==True:
                    print ('Set status to False -- Bad pixel in the central 3x3 cutout! %s'%(imgpath.split('/')[-1]))
            else:
                self.status = True
        
        self.obsjd = hd['OBSJD']
        self.zp = hd['MAGZP']
        self.e_zp = hd['MAGZPUNC']
        self.filter = hd['FILTER'][4]
        self.gain = hd['GAIN']
        # self.gain = 6.2 # Frank use this number ... ?
        self.seeing = hd['SEEING']
        self.programid = hd['PROGRMID']
        self.fieldid = hd['FIELDID']
        if 'CCDID' in hd.keys():
            self.ccdid = hd['CCDID']
        elif 'CCD_ID' in hd.keys():
            self.ccdid = hd['CCD_ID']
        if 'QID' in hd.keys():
            self.qid = hd['QID']
        else:
            self.qid = 99
            if self.verbose==True:
                print ('Set qid = 99 since this keyword is not in header: %s'%(imgpath.split('/')[-1]))
        
        self.filterid = hd['FILTERID']
        self.moonra = hd['MOONRA']
        self.moondec = hd['MOONDEC']
        self.moonillf  = hd['MOONILLF']
        self.moonphas = hd['MOONPHAS']
        self.airmass = hd['AIRMASS']
        
        # load psf cutout
        psf_fn = fits.open(psfpath)[0].data[12-r_psf:12+r_psf+1, 12-r_psf:12+r_psf+1]
        self.psf_fn = psf_fn  
        # upsample PSF using Gaussian interpolation
        self.upsample_psf_fn()
        
        # print out infomation
        if self.verbose == True:
            print ('processing record for %s'%self.imgpath)
            print ('\t gain=%.2f, diff_zp = %.4f'%(self.gain, self.zp))
            
            
    def upsample_psf_fn(self):
        '''
        length = pobj.length
        stampupsamplefac = pobj.stampupsamplefac
        psf_fn = pobj.psf_fn
        '''
        length = self.length
        psf_fn = self.psf_fn
        stampupsamplefac = self.stampupsamplefac
        
        # regularize input difference-image PSF (replace negative pixels with zero);
        ix = psf_fn < 0.
        psf_fn[ix] = 0.
        
        # newlength = stampupsamplefac * length
        upstep = 1./stampupsamplefac
        
        x_coarse, y_coarse = np.mgrid[0:length, 0:length]
        x_fine, y_fine = np.mgrid[0:length:upstep, 0:length:upstep]
        
        rbfi = Rbf(x_coarse.ravel(), y_coarse.ravel(), psf_fn.ravel(), function="gaussian")
        # interpolated psf frame
        fine_psf_fn = rbfi(x_fine.ravel(), y_fine.ravel()).reshape([x_fine.shape[0], 
                                                                    y_fine.shape[0]])
        # renormalize to unity
        fine_psf_fn /= np.sum(fine_psf_fn)
        # in the case that r_psf < 12
        fine_psf_fn *= np.sum(psf_fn) 
        self.fine_psf_fn = fine_psf_fn
        # norm = ImageNormalize(stretch=SqrtStretch())
        # plt.imshow(psf_fn, norm = norm)
        # norm2 = ImageNormalize(stretch=SqrtStretch())
        # plt.imshow(interp_psf_fn*25, norm=norm)
            
        
    def load_source_cutout(self):
        '''
        imgpath = pobj.imgpath
        pixX = pobj.pixX
        pixY = pobj.pixY        
        pixXint = pobj.pixXint
        pixYint = pobj.pixYint
        #bad_threshold = pobj.bad_threshold
        n_dty = pobj.n_dty
        n_dtx = pobj.n_dtx
        r_psf = pobj.r_psf
        length = pobj.length
        '''
        imgpath = self.imgpath        
        pixX = self.pixX
        pixY = self.pixY        
        pixXint = self.pixXint
        pixYint = self.pixYint
        #bad_threshold = self.bad_threshold
        n_dty = self.n_dty
        r_psf = self.r_psf
        length = self.length
        stampupsamplefac = self.stampupsamplefac
        newlength = self.newlength
        # n_dtx = self.n_dtx
        
        dt = fits.open(imgpath)[1].data  
        if (pixYint + r_psf + 2) > n_dty:
            new_patch = np.zeros((10, dt.shape[1]))
            dt = np.vstack([dt, new_patch])
        
        scr_fn_1 = dt[pixYint - r_psf - 1 : pixYint + r_psf + 2, 
                      pixXint - r_psf - 1 : pixXint + r_psf + 2]
        xoff_tobe = pixX - pixXint
        yoff_tobe = pixY - pixYint
        scr_fn_ = ndimage.shift(scr_fn_1, [-yoff_tobe, -xoff_tobe], order=3, 
                                mode='reflect', cval=0.0, prefilter=True)
        scr_fn = scr_fn_[1:-1, 1:-1] 
        
        # upsample input difference image stamp by simply rebinning
        # and ensuring sum of pixel fluxes is conserved.
        fine_scr_fn = scr_fn.repeat(stampupsamplefac, axis = 0).repeat(stampupsamplefac, axis = 1)
        fine_scr_fn /= stampupsamplefac**2
        # assert np.sum(fine_scr_fn)==np.sum(scr_fn)
        
        # norm = ImageNormalize(stretch=SqrtStretch())
        # plt.imshow(scr_fn, norm = norm)
        # norm2 = ImageNormalize(stretch=SqrtStretch())
        # plt.imshow(fine_scr_fn*25, norm=norm)
        
        bad_mask = np.isnan(scr_fn)
        nbad = np.sum(bad_mask)
        self.nbad = nbad
        self.scr_fn = scr_fn
        self.fine_scr_fn = fine_scr_fn
        
        if fine_scr_fn.shape[0]!=newlength or fine_scr_fn.shape[1]!=newlength:
            self.status = False
        
        if nbad!=0 and self.verbose==True:
            print ('%d bad pixels in %d*%d source frame' %(nbad, length, length))
        
        
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
        pixel =  np.array([[pixX_cor, pixY_cor]], np.float_)
        newcrd = w.wcs_pix2world(pixel, 0)
        ra_cor = newcrd[0][0] 
        dec_cor = newcrd[0][1]
        self.ra_cor = ra_cor
        self.dec_cor = dec_cor
        
        
    def load_bkg_cutout(self, manual_mask=False, col_mask_start=0, col_mask_end=0,
                        row_mask_start=0, row_mask_end=0):
        '''       
        Only need to manually mask the background if the number of bad 
        pixels in the background annulus is more than half of the total;
        Otherwise *median absolute deviation*(or percentiles) should give a 
        robust estimate of the background noise.
        
        imgpath = pobj.imgpath
        pixX = pobj.pixX
        pixY = pobj.pixY  
        # bad_threshold = pobj.bad_threshold
        r_bkg_in = pobj.r_bkg_in
        r_bkg_out = pobj.r_bkg_out
        '''
        imgpath = self.imgpath              
        pixX = self.pixX
        pixY = self.pixY
        # bad_threshold = self.bad_threshold
        r_bkg_in = self.r_bkg_in
        r_bkg_out = self.r_bkg_out
        
        dt = fits.open(imgpath)[1].data        
        positions = [(pixX, pixY)]
        annulus_aperture = CircularAnnulus(positions, 
                                           r_in = r_bkg_in, r_out = r_bkg_out)
        annulus_masks = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_masks[0].multiply(dt)
        
        bkg_fn = deepcopy(annulus_data)
        bad_bkg_mask = np.isnan(annulus_data)
        if manual_mask == True:    
            bad_bkg_mask[r_bkg_out+row_mask_start:r_bkg_out+row_mask_end, 
                         r_bkg_out+col_mask_start:r_bkg_out+col_mask_end] = True
        nbad_bkg = np.sum(bad_bkg_mask)
            
        self.nbad_bkg = nbad_bkg
        
        setnan = annulus_masks[0].data==0
        bkg_fn[setnan] = np.nan
        # bkgstd = np.nanstd(bkg_fn) 
        
        temp = bkg_fn.ravel()
        temp = temp[~np.isnan(temp)]
        bkgstd = 0.5 * (np.percentile(temp, 84.13)-np.percentile(temp, 15.86))
        # bkgstd = np.median(abs(temp - np.median(temp)))
        bkgmed = np.median(temp)
        
        self.bkgstd = bkgstd 
        self.bkg_fn = bkg_fn
        self.bkgmed = bkgmed
        
        if self.verbose == True:
            print ('\t bkgstd pixel RMS in original diff-image cutout = %.2f DN'%(self.bkgstd))
            print ('\t bkgmed pixel in original diff-image cutout = %.2f DN'%(self.bkgmed))
        
        
    def get_fine_scr_cor_fn(self):
        '''
        fine_scr_fn = pobj.fine_scr_fn
        fine_psf_fn = pobj.fine_psf_fn
        gain = pobj.gain
        bkgstd = pobj.bkgstd
        bkgmed = pobj.bkgmed
        stampupsamplefac = pobj.stampupsamplefac
        '''
        fine_scr_fn = self.fine_scr_fn
        fine_psf_fn = self.fine_psf_fn
        gain = self.gain
        bkgstd = self.bkgstd
        bkgmed = self.bkgmed
        stampupsamplefac = self.stampupsamplefac
            
        bkgmedupsamp = bkgmed/stampupsamplefac**2
        fine_scr_cor_fn = deepcopy(fine_scr_fn) - bkgmedupsamp
        fine_bad_mask = np.isnan(fine_scr_cor_fn)
        
        #--------
        # compute variance map for upsampled diff-image pixels used for photometry.
        bkgstdupsamp = bkgstd/stampupsamplefac
        fine_scr_cor_pos_fn = deepcopy(fine_scr_cor_fn)
        ix= fine_scr_cor_pos_fn < 0.
        fine_scr_cor_pos_fn[ix] = 0
        
        fine_scr_cor_var_fn = fine_scr_cor_pos_fn/gain + bkgstdupsamp**2
        
        self.fine_scr_cor_fn = fine_scr_cor_fn
        self.fine_bad_mask = fine_bad_mask
        self.nbadfine = np.sum(fine_bad_mask)
        
        _fine_scr_cor_ravel = fine_scr_cor_fn[~fine_bad_mask]
        _yerrsq = fine_scr_cor_var_fn[~fine_bad_mask]
        _yerr = np.sqrt(_yerrsq)
        self.yerrs = _yerr    
        self.y = _fine_scr_cor_ravel
        self.x = fine_psf_fn[~fine_bad_mask]
    
        
    def fit_psf(self):
        '''
        x = pobj.x
        y = pobj.y
        yerrs = pobj.yerrs 
        stampupsamplefac = pobj.stampupsamplefac
        '''
        x = self.x
        y = self.y
        yerrs = self.yerrs 
        stampupsamplefac = self.stampupsamplefac

        # one-parameter fit 
        Fpsf, eFpsf, apsf = mylinear_fit(x, y, yerrs, npar = 1)
        '''
        plt.errorbar(x, y, yerrs, fmt='.k')
        plt.plot(x, Fpsf*x, 'r-')
        '''
        
        #--------
        # compute reduced chi-square for PSF-fit.
        chi2 = (y - Fpsf*x)**2/yerrs**2
        chi2_red = np.sum(chi2) / (len(x)/stampupsamplefac**2-1)
        
        self.Fpsf = Fpsf
        self.eFpsf = eFpsf
        self.Fap = np.sum(y)/np.sum(x)
        self.chi2_red = chi2_red
        if self.verbose == True:
            print ('\t Fpsf = %.2f DN, eFpsf = %.2f DN, chi2_red = %.2f'%(self.Fpsf, self.eFpsf, self.chi2_red))
        
        
    def plot_cutouts(self, savepath=None):
        '''
        x = pobj.x
        y = pobj.y
        Fpsf = pobj.Fpsf
        fine_scr_cor_fn = pobj.fine_scr_cor_fn
        fine_psf_fn= pobj.fine_psf_fn
        Fpsf = pobj.Fpsf
        eFpsf = pobj.eFpsf
        fine_bad_mask = pobj.fine_bad_mask
        filtername = pobj.filter
        bkg_fn = pobj.bkg_fn
        seeing = pobj.seeing
        # length = pobj.length
        yerrs = pobj.yerrs
        chi2_red = pobj.chi2_red
        stampupsamplefac = pobj.stampupsamplefac
        '''
        cmap_name = 'viridis'
        
        # scr_fn = self.scr_fn
        x = self.x
        y = self.y
        Fpsf = self.Fpsf
        fine_scr_cor_fn = self.fine_scr_cor_fn
        fine_psf_fn= self.fine_psf_fn
        Fpsf = self.Fpsf
        eFpsf = self.eFpsf
        # fine_bad_mask = self.fine_bad_mask
        filtername = self.filter
        bkg_fn = self.bkg_fn
        seeing = self.seeing
        # length = self.length
        yerrs = self.yerrs
        chi2_red = self.chi2_red
        stampupsamplefac = self.stampupsamplefac
        
        model_fn = fine_psf_fn*Fpsf
    
        fig, ax = plt.subplots(4, 4, figsize=(9, 9))
        matplotlib.rcParams.update({'font.size': 15})
        '''
        norm = ImageNormalize(stretch=SqrtStretch())
        if np.sum(bad_mask) != 0:
            ax[0,0].imshow(scr_fn, cmap = cmap_name, origin='lower', norm=norm)
            ax[0,0].set_title('Unmasked, '+filtername, fontsize=15)
        else:
            ax[0,0].set_axis_off()
        '''
        norm2 = ImageNormalize(stretch=SqrtStretch())
        ax[0,0].imshow(fine_scr_cor_fn, cmap = cmap_name, origin='lower', norm=norm2)
        ax[0,0].set_title('Data, '+filtername, fontsize=15)
        ax[0,1].imshow(model_fn, cmap = cmap_name, origin='lower', norm=norm2)
        ax[0,1].set_title('PSF model', fontsize=15)
        normnew = ImageNormalize(stretch=SqrtStretch())
        ax[0,3].imshow(bkg_fn, cmap = cmap_name, origin='lower', norm=normnew)
        ax[0,3].set_title('Background', fontsize=15)
        
        norm1 = ImageNormalize(stretch=SqrtStretch())
        ax[0,2].imshow(fine_scr_cor_fn-model_fn, cmap = cmap_name, origin='lower', norm=norm1)
        ax[0,2].set_title('Residual', fontsize=15)
        ax[0][0].set_xticklabels([])
        ax[0][0].set_yticklabels([])
        ax[0][1].set_xticklabels([])
        ax[0][1].set_yticklabels([])
        ax[0][2].set_xticklabels([])
        ax[0][2].set_yticklabels([])
        ax[0][3].set_xticklabels([])
        ax[0][3].set_yticklabels([])
        ax[0,0].tick_params(axis='both', which='both', direction='in')
        ax[0,1].tick_params(axis='both', which='both', direction='in')
        ax[0,2].tick_params(axis='both', which='both', direction='in')
        ax[0,3].tick_params(axis='both', which='both', direction='in')
        
        ax4 = plt.subplot2grid((4, 1), (1, 0), rowspan=2)
        ax4.errorbar(x, y, yerrs/stampupsamplefac, fmt='.k', zorder=1)
        xx = np.array([np.min(x), np.max(x)])
        ax4.plot(xx, Fpsf*xx, 'r-', zorder=2)
        ax4.tick_params(axis='both', which='both', direction='in')
        # ax4.set_xlim(0-1, length**2+1)

        ylims = ax4.get_ylim()
        ax4.set_xticklabels([])
        yloc1 = ylims[0] + (ylims[1] - ylims[0])*0.9
        plt.text(xx.mean() ,yloc1, 'flux = %.1f, e_flux = %.1f'%(Fpsf, eFpsf), fontsize=15, color='m')
        yloc2 = ylims[0] + (ylims[1] - ylims[0])*0.8
        plt.text(xx.mean() ,yloc2, 'seeing = %.3f'%seeing, fontsize=15, color='m')
        yloc3 = ylims[0] + (ylims[1] - ylims[0])*0.1
        plt.text(xx.mean() ,yloc3, 'chi2_red = %.3f'%chi2_red, fontsize=15, color='m')
        
        ax5 = plt.subplot2grid((4, 1), (3, 0))
        ax5.errorbar(x, y -Fpsf*x, yerrs/stampupsamplefac, fmt='.k', zorder=1)
        plt.plot(xx, [0,0], color='grey', linewidth = 2, alpha= 0.5, zorder=2)
        ax5.tick_params(axis='both', which='both', direction='in')
        
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()