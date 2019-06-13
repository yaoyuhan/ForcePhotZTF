#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:09:21 2019

@author: yuhanyao
"""
import numpy as np
from astropy.table import Table
import emcee
import time
import scipy.optimize as op
from multiprocessing import Pool

# =========================================================================== #
# define MCMC functions
def systematic_lnlike(theta, x, y, sigma_y):
    m, lnsig_0 = theta
    model = m * x
    sig_0 = np.exp(lnsig_0)
    
    chi2_term = -1/2*np.sum((y - model)**2/(sigma_y**2 + sig_0**2))
    error_term = np.sum(np.log(1/np.sqrt(2*np.pi*(sigma_y**2 + sig_0**2))))
    ln_l = chi2_term + error_term
    
    return ln_l

systematic_nll = lambda *args: -systematic_lnlike(*args)

def systematic_lnprior(theta):
    m, lnsig_0 = theta
    if (-1e6 < m < 1e6 and -10 < lnsig_0 < 10):
        return 0.0
    return -np.inf

# The full log-probability function is
def systematic_lnprob(theta, x, y, yerr):
    lp = systematic_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + systematic_lnlike(theta, x, y, yerr)


def pool_sys_process(df, i):
    subdf = df.iloc[np.where(df['index']==i)]
    x = subdf['x'].values
    y = subdf['y'].values
    yerr = subdf['ey'].values
    
    result = op.minimize(systematic_nll, [0, 1],
                         method='Powell', args=(x, y, yerr))
    ml_guess = result["x"]
                         
    if ml_guess[-1] < -9.5:
        ml_guess[-1] = -9
    ndim = len(ml_guess)
    nwalkers = 250
    
    pos = [ml_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]

    # set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, systematic_lnprob, 
                                    args=(x, y, yerr))

    max_samples = 50000

    index = 0
    autocorr = np.empty(max_samples)
    old_tau = np.inf
    for sample in sampler.sample(pos, iterations=max_samples):
        if ((sampler.iteration % 250) and 
            (sampler.iteration < 5000)):
            continue
        elif ((sampler.iteration % 1000) and 
              (5000 <= sampler.iteration < 15000)):
            continue
        elif ((sampler.iteration % 2500) and 
              (15000 <= sampler.iteration)):
            continue
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[sampler.iteration-1] = np.mean(tau)
        index += 1

        # Check convergence
        
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
        
    if np.isnan(tau[0])==True or np.isinf(tau[0])==True:
        print ('index = %d, not enough epochs to converge'%(i))
        samples = sampler.get_chain(discard=1000, flat=True)
        print (tau)
    else:
        samples = sampler.get_chain(discard=int(10*tau[0]), flat=True)
        
    Fmcmc_sigmas = np.percentile(samples[:,0], (0.13, 2.27, 15.87, 50, 84.13, 97.73, 99.87))
    
    print ('index = %d, Fmcmc_med = %.2f'%(i, Fmcmc_sigmas[3]))
    result = np.hstack([np.array([i]), # position argument
                        Fmcmc_sigmas])
    return result


def get_forced_phot_mcmc(name, targetdir, ncpu, verbose=False):
    '''Perform MCMC fit to PSF model to produce forced phot
    '''
    
    print ("Start fitting PSF using MCMC method for %s"%(name))
    info_file = targetdir + 'lightcurves/force_phot_{}_info.fits'.format(name)
    xy_file = targetdir + 'lightcurves/xydata_{}.fits'.format(name)

    info_tbl = Table.read(info_file)
    xy_tbl = Table.read(xy_file)
    info_df = info_tbl.to_pandas()
    xy_df = xy_tbl.to_pandas()
    
    
    pool = Pool(ncpu)
    tstart = time.time()
    results = [pool.apply_async(pool_sys_process, args=(xy_df, i,)) 
                for i in xy_df['index'].unique()]
    output = [p.get() for p in results]
    tend = time.time()

    print("Pool map took {:.4f} sec".format(tend-tstart))
    
    output_arr = np.array(output)
    
    # test that ordering is identical
    if np.all(info_df['diffimgname'] == xy_df['path'].unique()):
        Fmcmc = np.zeros_like(info_df['zp'])
        Fmcmc_low_1sigma = np.zeros_like(Fmcmc)
        Fmcmc_low_2sigma = np.zeros_like(Fmcmc)
        Fmcmc_low_3sigma = np.zeros_like(Fmcmc)
        Fmcmc_high_1sigma = np.zeros_like(Fmcmc)
        Fmcmc_high_2sigma = np.zeros_like(Fmcmc)
        Fmcmc_high_3sigma = np.zeros_like(Fmcmc)
        for res_idx in output_arr[:,0].astype(int):
            idx = np.where(xy_df['index'].unique() == res_idx)[0]
            Fmcmc_low_3sigma[idx] = output_arr[res_idx, 1]
            Fmcmc_low_2sigma[idx] = output_arr[res_idx, 2]
            Fmcmc_low_1sigma[idx] = output_arr[res_idx, 3]
            Fmcmc[idx] = output_arr[res_idx, 4]
            Fmcmc_high_1sigma[idx] = output_arr[res_idx, 5]
            Fmcmc_high_2sigma[idx] = output_arr[res_idx, 6]
            Fmcmc_high_3sigma[idx] = output_arr[res_idx, 7]
    else:
        raise ValueError("Input files do not have the same order")
    
    # calculate flux
    Fmcmc_unc = (Fmcmc_high_1sigma-Fmcmc_low_1sigma)/2.
    
    f0 = 10**(info_df['zp'].values/2.5)
    f0_unc = f0 / 2.5 * np.log(10) * info_df['ezp']
    Fratio =  Fmcmc/f0
    Fratio_unc = np.hypot(Fmcmc_unc/f0, Fmcmc*f0_unc/f0**2)
    
    info_df['Fmcmc'] = Fmcmc
    info_df['Fmcmc_unc'] = Fmcmc_unc
    info_df['Fmcmc_low_3sigma'] = Fmcmc_low_3sigma
    info_df['Fmcmc_low_2sigma'] = Fmcmc_low_2sigma
    info_df['Fmcmc_low_1sigma'] = Fmcmc_low_1sigma
    info_df['Fmcmc_high_3sigma'] = Fmcmc_high_3sigma
    info_df['Fmcmc_high_2sigma'] = Fmcmc_high_2sigma
    info_df['Fmcmc_high_1sigma'] = Fmcmc_high_1sigma
    info_df['Fratio'] = Fratio
    info_df['Fratio_unc'] = Fratio_unc

    info_df.to_hdf(targetdir + 'lightcurves/{}_force_phot_nob.h5'.format(name), 'lc')

