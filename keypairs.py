#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:33:27 2019

@author: yuhanyao
"""
import os


def get_keypairs():
    # print (os.path.dirname(os.path.realpath(__file__)) )
    
    with open(os.path.join(os.path.dirname(__file__), 'auth_marshal.txt'),'r') as f:
        lines =  f.readlines()
        words = lines[0].split(':')
        DEFAULT_AUTH_marshal = (words[0], words[1].split('\n')[0])
        f.close()
    
    with open(os.path.join(os.path.dirname(__file__), 'auth_kowalski.txt'),'r') as f:
        lines =  f.readlines()
        words = lines[0].split(':')
        DEFAULT_AUTH_kowalski = (words[0], words[1].split('\n')[0])
        f.close()
    
    with open(os.path.join(os.path.dirname(__file__), 'auth_ipac.txt'),'r') as f:
        lines =  f.readlines()
        words = lines[0].split(':')
        DEFAULT_AUTH_ipac = (words[0], words[1].split('\n')[0])
        f.close()
    
    with open(os.path.join(os.path.dirname(__file__), 'auth_fps.txt'),'r') as f:
        lines =  f.readlines()
        words = lines[0].split(':')
        DEFAULT_AUTH_fps = (words[0], words[1].split('\n')[0])
        f.close()
        
    return [DEFAULT_AUTH_marshal, DEFAULT_AUTH_kowalski, DEFAULT_AUTH_ipac, DEFAULT_AUTH_fps]