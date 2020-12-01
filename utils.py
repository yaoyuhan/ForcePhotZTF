#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import gzip
import glob
import time
import pandas as pd
import numpy as np
import subprocess
import astropy.units as u
from astropy.io import fits
import astropy.io.ascii as asci
from astropy.time import Time

from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.simplefilter('ignore', category = AstropyDeprecationWarning)


def database_query(s, q, nquery = 5):
    """
    To prevent error in Kowalski query:
    KeyError: 'result_data'
    """
    r = {}
    cnt = 0
    while cnt < nquery:
        r = s.query(query=q)
        if "data" in r:
            break
        time.sleep(5)        
        cnt = cnt + 1
    return r




def get_dets(s, name):
    q = {"query_type": "find",
         "query": {
             "catalog": "ZTF_alerts",
             "filter": {
                     'objectId': {'$eq': name},
                     'candidate.isdiffpos': {'$in': ['1', 't']},
             },
             "projection": {
                     "_id": 0,
                     "candidate.jd": 1,
                     "candidate.ra": 1,
                     "candidate.dec": 1,
                     "candidate.magpsf": 1,
                     "candidate.sigmapsf": 1,
                     "candidate.fid": 1,
                     "candidate.programid": 1,
                     "candidate.drb": 1
             }
         }  
         }  
    query_result = database_query(s, q, nquery = 10)
    out = query_result['data']
    return out
