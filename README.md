Copyright (C) 2018 Juan D. Soler.

AstroHOG was created by Juan D. Soler, but is now maintained by several people. 
See the AstroHOG repository on GitHub for further details.

AstroHOG
==================================

AstroHOG is set of tools for the comparison of extended spectral-line observations (PPV cubes).
In essence, the histrogram of oriented gradients (HOG) technique takes as input two PPV cubes and provides an estimate of their spatial correlation across velocity channels.

If you make use of this package in a publication, please cite our accompanying [paper](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1809.08338):

```
@ARTICLE{2018arXiv180908338S,
   author = {{Soler}, J.~D. and {Beuther}, H. and {Rugel}, M. and {Wang}, Y. and 
	{Anderson}, L.~D. and {Clark}, P.~C. and {Glover}, S.~C.~O. and 
	{Goldsmith}, P.~F. and {Goodman}, A. and {Hennebelle}, P. and 
	{Henning}, T. and {Heyer}, M. and {Kainulainen}, J. and {Klessen}, R.~S. and 
	{McClure-Griffiths}, N.~M. and {Menten}, K.~M. and {Mottram}, J.~C. and 
	{Ragan}, S.~E. and {Schilke}, P. and {Smith}, R.~J. and {Urquhart}, J.~S. and 
	{Bigiel}, F. and {Roy}, N.},
    title = "{Histogram of oriented gradients: a technique for the study of molecular cloud formation}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1809.08338},
 keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
     year = 2018,
    month = sep,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180908338S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Citation courtesy of [ADS](https://ui.adsabs.harvard.edu/#).

=================================
Dependencies:

astropy
numpy
sys
scipy
nose
scikit-image
tqdm
pycircstat
pandas
reproject

================================

First steps: 
Check out the example notebooks: examples/compareimages.ipynb and examples/comparecubes.ipynb 
