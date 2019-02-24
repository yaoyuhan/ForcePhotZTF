# ForcePhotZTF
This is written for getting the light curves of ZTF transients by forced-PSF photometry. 

## Python versions and package dependencies:
- python 3.6
- numpy==1.15.1
- scipy=1.1.0
- pandas=0.23.4
- matplotlib==2.2.3
- ztfquery==1.2.6
- photutils=0.5
- image_registration=0.2.4

## Provide your username and password before using this package:
ForcePhotZTF performs multiple tasks including:
- downloading images from IPAC
- getting ZTF light curves from the GROWTH Marshal website (Kasliwal et al. 2019)
- getting alert package info from Kowalski (optional)
- running forced photometry on Caltech forced phometry service (fps, Masci et al. 2019), kindly provided by Frank Masci at IPAC. I've compared his result and my ForcePhotZTF result -- they are very consistent with each other. The only difference may be the size of cutout and the assumption in gain factor. Thus, this step is optional. I only want some column information given in the header of sci images, which can also be obtained from fps. 

You need to provide your username and password (save them to a file) in the format of 'username:password\n'. Prepare four txt files:\\
1. auth_fps.txt <br>
(If you do know have the access, just print ' : \n' to this file and do not run the final step shown in the demo)
2. auth_ipac.txt
3. auth_kowalski.txt <br>
(This is not necessary in any of the main steps. Thus, just print ' : \n' to this file if you do not have Kowalski access)
4. auth_marshal.txt 

## How to use this package.
Please see demo.ipynb

## Scripts not relavant to forced-PSF photometry, but might be useful:
#### reference.py 

This is a demo of how to download marshal lightcurves without using ztfquery, and get reference image epochs from Kowalski or IPAC. Reference image epochs are important since you may want to make sure that no supernova light is in the reference images.

#### snid_run.py

This is a demo of how to automatically run SNID on Caltech computer pharos (and thus written in python 2). 
Please see here https://people.lam.fr/blondin.stephane/software/snid/howto.html for other options.
