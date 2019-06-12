# ForcePhotZTF
This is written for getting the light curves of ZTF transients by forced-PSF photometry. </br>
If you wish to use light curves produced by this code please offer coauthorship to the contributors of this package.</br>

Please cite our paper Yao et al. (2019, in prep).

## Python versions and package dependencies:
- python 3.6
- numpy 1.16.2
- scipy 1.2.1
- pandas 0.24.1
- emcee 3.0rc2 [link](https://emcee.readthedocs.io/en/latest/user/install/)
- matplotlib 3.0.3
- astropy 3.1.2
- sncosmo 1.7.1
- ztfquery 1.4.0
- photutils 0.6
- coner 2.0.1
- image_registration 0.2.4
- penquins 1.0.0 [link to Kowalski](https://github.com/dmitryduev/kowalski)

## Provide your username and password before using this package:
ForcePhotZTF performs multiple tasks including:
- downloading images from IPAC
- getting ZTF light curves from the GROWTH Marshal website (Kasliwal et al. 2019)
- getting alert package info from Kowalski (optional)
- running forced photometry on Caltech forced phometry service (fps, Masci et al. 2019), kindly provided by Frank Masci at IPAC. I've compared his result and my ForcePhotZTF result -- they are very consistent with each other. The only difference may be the size of cutout and the assumption in gain factor. Thus, this step is optional. I only want some column information given in the header of sci images, which can also be obtained from fps. 

You need to provide your username and password (save them to a file) in the format of `username:password`. Prepare four txt files:
1. auth_fps.txt <br>
(If you do not have the access, just print `:` to this file and do not run the final step shown in the demo)
2. auth_ipac.txt
3. auth_kowalski.txt <br>
(This is not necessary in any of the main steps. Thus, just print `:` to this file if you do not have Kowalski access)
4. auth_marshal.txt 

## How to use this package.
Please see demo.ipynb
