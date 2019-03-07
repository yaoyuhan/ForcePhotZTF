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
    directory=os.getcwd()+'/data/'
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