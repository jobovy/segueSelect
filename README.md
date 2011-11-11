#segueSelect

##AUTHOR

Jo Bovy - bovy at ias dot edu

If you find this code useful in your research, please cite
[arXiv:111.1724](http://arxiv.org/abs/1111.1724). Thanks!


##INSTALLATION

Standard python setup.py build/install


##DEPENDENCIES

This package requires [NumPy](http://numpy.scipy.org/), [Scipy] (http://www.scipy.org/), [Matplotlib] (http://matplotlib.sourceforge.net/), and [Pyfits](http://www.stsci.edu/resources/software_hardware/pyfits). To use coordinate transformations, [galpy](https://github.com/jobovy/galpy) is required.


##DOCUMENTATION

segueSelect is a Python package that implements the model for the
SDSS/SEGUE selection function described in Appendix A of
[arXiv:111.1724](http://arxiv.org/abs/1111.1724). 

To get started, download the files at
http://sns.ias.edu/~bovy/segueSelect/, put them in some directory,
untar the segueplates.tar.gz file (tar xvzf segueplates.tar.gz) and
define an environment variable SEGUESELECTDIR that points to this
directory (*without* trailing slash).

After installing the package (python setup.py install) you can use the
package as

	rom segueSelect import segueSelect
	selectionFunction= segueSelect(sample='G',sn=15,select='all')

to get the selection function for the SEGUE G star sample, using a
signal-to-noise cut of 15, and selection all stars in the G star color
range (as opposed to select='program', which just uses the stars that
were targeted as G stars).

The selection function is determined on the fly, so sample selection
can be adjusted if desired. Relevant options are

    ug= if True, cut on u-g, (default: False)
    	if list/array cut to ug[0] < u-g< ug[1]
    ri= if True, cut on r-i,  (default: False)
    	if list/array cut to ri[0] < r-i< ri[1]
    sn= if False, don't cut on SN, 
    	if number cut on SN > the number (default: 15)
    ebv= if True, cut on E(B-V), 
    	 if number cut on EBV < the number (default: 0.3)

The type of selection function can be set separately for 'bright'
plates and 'faint' plates. The default for both is 'tanhrcut', which
is the selection function described in Appendix A of Bovy et
al. (2011), but other options include:

    type_bright= or type_faint=
    
        'constant': constant for each plate up to the faint limit for
	the sample (decent for bright plates, *bad* for faint plates
	that never reach as far as the faint limit) 

	'sharprcut': use a sharp cut rather than a hyperbolic tangent
	cut-off at the faint end of the apparent magnitude range

The recommended setting is 'tanhrcut' for both bright and faint plates.


Once the selection function is initialized it can be evaluated as

     late=1880
     value= selectionFunction(plate,r=16.)

where value is then the fraction of stars in the SEGUE spectroscopic
sample for that plate number and that r-band apparent magnitude.

