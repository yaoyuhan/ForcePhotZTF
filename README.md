# ForcePhotZTF
This is written for getting the light curves of ZTF transients by force photometry.
Please see demo.ipynb for how to use this package and the pdf file for the methodology.

Still under developmemt. Suggestions are welcomed!

## Python versions and package dependencies:
- python 3.6
- numpy==1.15.1
- scipy=1.1.0
- pandas=0.23.4
- matplotlib==2.2.3
- ztfquery==1.2.6
- photutils=0.5
- image_registration=0.2.4

## Scripts not relavant to forced-PSF photometry, but might be useful:
#### reference.py 

This is a demo of how to download marshal lightcurves without using ztfquery, and get reference image epochs from Kowalski or IPAC. Reference image epochs are important since you may want to make sure that no supernova light is in the reference images.

#### snid_run.py

This is a demo of how to automatically run SNID on Caltech computer pharos (and thus written in python 2). 
Please see here https://people.lam.fr/blondin.stephane/software/snid/howto.html for other options.
