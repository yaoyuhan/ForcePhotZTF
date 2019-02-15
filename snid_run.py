#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:55:58 2019

@author: yuhanyao
"""
import os
import glob, shutil, subprocess
import numpy as np


pathtb = '/home/yyao/name161_z.txt'
specdir = '/home/yyao/spectra/'

f =  open(pathtb, 'r')
lines = f.readlines()
f.close()

ztfnames = [x[2:14] for x in lines[1:]]
z_adopt = [float(x.split('\n')[0][15:]) for x in lines[1:]]

for i in range(len(ztfnames)):
    name = ztfnames[i]
    try:
        os.stat(specdir+name+'/')
    except:
        os.mkdir(specdir+name+'/')
    os.chdir(specdir+name+'/')
    fps = glob.glob(specdir+name+'*.ascii')
    fps = np.array(fps)
    ind = np.ones(len(fps), dtype=bool)
    prestrings = []
    for j in range(len(fps)):
        fpsnow = fps[j]
        strnow = fpsnow.split('.ascii')[0][:-3]
        if strnow not in prestrings:
            prestrings.append(strnow)
        else:
            ind[j] = False
    fps = fps[ind]
    if len(fps)==0:
        1+1
    else:
        print name
        print z_adopt[i]
        for x in fps:
            specfile = specdir+name+'/'+x.split('/')[-1]
            shutil.copyfile(x, specdir+name+'/'+x.split('/')[-1])
            if z_adopt[i]==-999:
                subprocess.call('/scr2/nblago/software/snid-5.0/snid verbose=0 plot=0 fluxout=5 '+specfile, shell=True)
            else:
                #zarg = 'forcez=%.6f'%z_adopt[i]
                subprocess.call('/scr2/nblago/software/snid-5.0/snid forcez=%.6f verbose=0 plot=0 fluxout=5 '%z_adopt[i] + specfile, shell=True)
