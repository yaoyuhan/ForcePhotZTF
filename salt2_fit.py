#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:22:05 2019

@author: yuhanyao
"""
import os
import sncosmo
import numpy as np


def add_ZTFfilters():
    directory='/Users/yuhanyao/Documents/ForcePhotZTF/data/'
    bandsP48 = {'p48i': 'P48_I.dat',
                'p48r': 'P48_R.dat',
                'p48g': 'P48_g.dat'}

    fileDirectory = directory+'filters/P48/'
    for bandName, fileName in bandsP48.items():
        filePath = os.path.join(fileDirectory, fileName)
        if not os.path.exists(filePath):
            raise IOError("No such file: %s" % filePath)
        b = np.loadtxt(filePath)
        band = sncosmo.Bandpass(b[:, 0], b[:, 1], name=bandName)
        sncosmo.registry.register(band, force=True)


# the following functions are partly borrowed from 
# https://gist.github.com/ufeindt/cf3a4dd6484f4e96aec1
def _get_bandmag(band, magsys, t=0, rest_frame=True, **kwargs):
    """
    Returns mag at max for the model, band and magsys
    
    Arguments:
        model  -- sncosmo model, e.g. SALT2
        band   -- sncosmo band object or string, e.g. 'bessellb'
        magsys -- magnitude system, e.g. 'ab'
        
    Keyword arguments:
        t -- time relative to t0 (observer-frame), at which to evaluate
        rest_frame -- default: True, overrides the redshifts
    """
    
    model = sncosmo.Model(source='salt2')
    if rest_frame:
        kwargs['z'] = 0
    
    model.set(**kwargs)
    return model.bandmag(band, magsys, kwargs['t0'] + t)


def _get_bandmag_gradient(band, magsys, param, sig, fixed_param,
                          t=0, rest_frame=True):
    """
    Return gradient of _get_bandmag as function of param
        
    param, sig must be dictionaries of means and uncertainties
    Best use odicts to make sure that the order of the components is correct
    """
    
    model = sncosmo.Model(source='salt2')
    out = []
    
    if rest_frame:
        if 'z' in param.keys():
            param['z'] = 0
        if 'z' in fixed_param.keys():
            fixed_param['z'] = 0

    model.set(**fixed_param)
    for key,val in param.items():
        model.set(**param)
        h = sig[key] / 100.
    
        model.set(**{key: val - h})
        m0 = model.bandmag(band, magsys, param['t0'] + t)
        
        model.set(**{key: val + h})
        m1 = model.bandmag(band, magsys, param['t0'] + t)
        
        out.append((m1 - m0) / (2. * h))
    
    return np.array(out)
