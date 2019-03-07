#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:13:18 2019

@author: yuhanyao

It is so dumb that Caltech private computer does not have iPython!
"""
import sys
import argparse


def add_function(a, b):
    return a+b


if __name__ == '__main__':
    
    a = sys.argv[1]
    b = sys.argv[2]
    result = add_function(a, b)
    print (result)