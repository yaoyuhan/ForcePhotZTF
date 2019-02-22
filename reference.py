#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:11:22 2019

@author: yuhanyao
"""
import os
import requests
import numpy as np
import pandas as pd

from ztfquery import query
from ztfquery import marshal
from penquins import Kowalski

import astropy.io.ascii as asci
from astropy.time import Time
from astropy.table import Table

name = 'ZTF18abcdefg'
from ForcePhotZTF.keypairs import get_keypairs
DEFAULT_AUTHs = get_keypairs()
DEFAULT_AUTH_marshal = DEFAULT_AUTHs[0]
DEFAULT_AUTH_kowalski = DEFAULT_AUTHs[1]
DEFAULT_AUTH_ipac = DEFAULT_AUTHs[2]


def download_marshal_lightcurve(name, DEFAULT_AUTH_marshal):
    '''
    alternatively you can use ztfquery
    '''
    # generate directory to store data
    cwd = os.getcwd()
    targetdir = cwd+'/'+name+'/'
    try:
        os.stat(targetdir)
    except:
        os.mkdir(targetdir)
        
    try:
        os.stat(targetdir+'lightcurves/')
    except:
        os.mkdir(targetdir+'lightcurves/')
        
    # marshal query
    r = requests.get('http://skipper.caltech.edu:8080/cgi-bin/growth/plot_lc.cgi?name='+name, 
                     auth=(DEFAULT_AUTH_marshal[0], DEFAULT_AUTH_marshal[1]))
    tables = pd.read_html(r.content)
    mtb = tables[14]
    mtb = mtb.drop([0], axis=1)
    mtb.to_csv(targetdir+'lightcurves/'+'/marshal_lc_'+name+'.csv',header=False, index = False)
    
    
def query_kowalski(name, DEFAULT_AUTH_kowalski):
    '''
    Please see 
    https://kowalski.caltech.edu/docs/python_client
    of description
    '''
    # generate directory to store data
    cwd = os.getcwd()
    targetdir = cwd+'/'+name+'/'
    try:
        os.stat(targetdir)
    except:
        os.mkdir(targetdir)
        
    try:
        os.stat(targetdir+'lightcurves/')
    except:
        os.mkdir(targetdir+'lightcurves/')
        
    # kowalski query
    k = Kowalski(username=DEFAULT_AUTH_kowalski[0], password=DEFAULT_AUTH_kowalski[1], verbose=False)
    
    qu = {"query_type": "general_search", 
          "query": "db['ZTF_alerts'].find({'objectId': {'$eq': '%s'}})"%name}
    r = k.query(query=qu)
    
    if 'result_data' in r.keys():
        rdata = r['result_data']
        rrdata = rdata['query_result']
        n = len(rrdata)
        jds = np.zeros(n)
        fids = np.zeros(n)
        jdstarts = np.zeros(n)
        jdends = np.zeros(n)
        fieldids = np.zeros(n)
        magpsfs = np.zeros(n)
        sigmapsfs = np.zeros(n)
        diffimfile = [' '*57 for x in range(n)]
            
        for i in range(n):
            rrr = rrdata[i]
            # yyao: please see rrr['candidate'].keys() for what infomation you can get
            # I only saved jd, filterid, fieldid, magpsf, sigmapsf, jdstartref, jdendref, and diffimfilename in this demo...
            jds[i] = rrr['candidate']['jd']
            fids[i] = rrr['candidate']['fid']
            fieldids[i] = rrr['candidate']['field']
            magpsfs[i] = rrr['candidate']['magpsf']
            sigmapsfs[i] = rrr['candidate']['sigmapsf']
            
            # yyao: sometimes these infomation is not in the keys (reference image not made?)
            if 'jdstartref' in rrr['candidate'].keys(): 
                jdstarts[i] = rrr['candidate']['jdstartref']
            if 'jdendref' in rrr['candidate'].keys(): 
                jdends[i] = rrr['candidate']['jdendref']
            if 'pdiffimfilename' in rrr['candidate'].keys(): 
                diffimfile[i] = rrr['candidate']['pdiffimfilename']
        tb = Table([jds, fids, fieldids, magpsfs, sigmapsfs, jdstarts, jdends, diffimfile], 
                   names = ['jd', 'fid', 'fieldid', 'magpsf', 'sigmapsf', 
                            'jdstartref', 'jdendref', 'diffimfilename'])
        tb.write(targetdir+'lightcurves/'+'/kowalski_lc_'+name+'.csv')
        
    else:
        print ('query is not succesful for %s'%name)
        # yyao: sometimes this is not successful. I don't know why.
        
        
def get_pos(name):
    """
    Get position of target, borrow from
    https://github.com/annayqho/FinderChart/blob/master/ztf_finder.py
    """
    m = marshal.MarshalAccess()
    m.load_target_sources()
    coords = m.get_target_coordinates(name)
    ra = coords.ra.values[0]
    dec = coords.dec.values[0]
    return ra, dec
        
        
def query_ipac(name):
    '''
    To query ipac for reference exposure epoch, you need to send the filterid, fieldid, ccdid, and qid
    This can firstly be downloaded from ipac using ztfquery
    
    Please see
    https://irsa.ipac.caltech.edu/TAP/sync?query=select+column_name,description,unit,ucd,utype,datatype,principal,indexed+from+TAP_SCHEMA.columns+where+table_name=%27ztf.ztf_current_meta_ref%27+order+by+column_index&format=html
    of the description of IPAC reference image columns
    '''
    
    # Step 0. As usual, generate directory to store data
    cwd = os.getcwd()
    targetdir = cwd+'/'+name+'/'
    try:
        os.stat(targetdir)
    except:
        os.mkdir(targetdir)
        
    try:
        os.stat(targetdir+'lightcurves/')
    except:
        os.mkdir(targetdir+'lightcurves/')
    
    # Step 1. ipac does not know ZTF name, so let's get the coordinate first
    ra1, dec1 = get_pos(name)
    np.savetxt(targetdir+'/coo_marshal.reg', [ra1, dec1])
    
    # Step 2. download infomation about all images that has ever covered this coordinate
    zquery = query.ZTFQuery()
    print("Querying for metadata...")
    # note: the unit of size is [degree], you may want to change it
    zquery.load_metadata(kind = 'sci', radec = [ra1, dec1], size = 0.003) 
    out = zquery.metatable
    final_out = out.sort_values(by=['obsjd'])
    final_out.to_csv(targetdir+'/irsafile.csv')
    
    # Step 3. get reference epoch for every row in the file `irsafile.csv`
    s = requests.Session()
    s.post('https://irsa.ipac.caltech.edu/account/signon/login.do?josso_cmd=login', 
            data={'josso_username': DEFAULT_AUTH_ipac[0], 'josso_password': DEFAULT_AUTH_ipac[1]})
    
    mylc = Table([final_out['field'].values, final_out['ccdid'].values, final_out['fid'].values,
                  final_out['qid'].values, final_out['obsjd'].values], 
                  names = ['field', 'ccdid', 'fid', 'qid', 'obsjd'])
    # For each unique ccd-quarant (fcqf id), we only want to query once to save time
    fcqf = mylc['field']*10000 + mylc['ccdid']*100 + mylc['qid']*10 + mylc['fid'] 
    mylc['fcqf'] = fcqf
    fcq_uniq = []
    for x in mylc['fcqf']:
        if x not in fcq_uniq:
            fcq_uniq.append(x)
    fcq_uniq = np.array(fcq_uniq)
    
    jdref_start = np.zeros(len(mylc))
    jdref_end = np.zeros(len(mylc))
        
    for j in range(len(fcq_uniq)):
        fcqnow = fcq_uniq[j]
        temp1 = fcqnow - fcqnow%10000
        fieldnow = np.int(temp1/10000)
        temp2 = fcqnow - temp1
        temp3 = temp2 - temp2%100
        ccdidnow = np.int(temp3/100)
        temp4 = temp2 - temp3
        qidnow = np.int((temp4 - temp4%10)/10)
        filteridnow = temp4 - qidnow*10
        if filteridnow==1:
            fltidnow = 'zg'
        elif filteridnow==2:
            fltidnow = 'zr'
        elif filteridnow==3:
            fltidnow = 'zi'

        url = 'https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE=field=' +\
                    '%d'%(fieldnow)+'%20AND%20ccdid='+'%d'%(ccdidnow) +\
                    '%20AND%20qid='+'%d'%(qidnow)+\
                    '%20AND%20filtercode=%27'+'%s'%(fltidnow)+'%27' 
        r = requests.get(url, cookies=s.cookies)
        stringnow = r.content
        stnow = stringnow.decode("utf-8")
        tbnowj = asci.read(stnow)
        if len(tbnowj)==0:
            print ('no reference image: fcqf id = %d'%fcqnow)
        else:
            t0 = tbnowj['startobsdate'].data.data[0]
            t1 = tbnowj['endobsdate'].data.data[0]
            tstart = Time(t0.split(' ')[0] + 'T' + t0.split(' ')[1][:-3], 
                          format='isot', scale='utc')
            tend = Time(t1.split(' ')[0] + 'T' + t1.split(' ')[1][:-3], 
                        format='isot', scale='utc')
        
            ind = mylc['fcqf'] == fcqnow
            jdref_start[ind] = tstart.jd
            jdref_end[ind] = tend.jd
            
    mylc['jdref_start'] = jdref_start
    mylc['jdref_end'] = jdref_end
    mylc.write(targetdir+'lightcurves/'+'/ipac_info_'+name+'.csv')