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
[arXiv:111.1724](http://arxiv.org/abs/1111.1724). It automatically
determines the selection fraction as a continuous function of apparent
magnitude for each plate. The selection function can be determined for
any desired sample cuts in signal-to-noise ratio, u-g, r-i, and
E(B-V).

To get started, download the files at
http://sns.ias.edu/~bovy/segueSelect/, put them in some directory,
untar the segueplates.tar.gz file (tar xvzf segueplates.tar.gz) and
define an environment variable SEGUESELECTDIR that points to this
directory (*without* trailing slash).

After installing the package (python setup.py install) you can use the
package as

	from segueSelect import segueSelect
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

     plate=1880
     value= selectionFunction(plate,r=16.)

where value is then the fraction of stars in the SEGUE spectroscopic
sample for that plate number and that r-band apparent magnitude.


##ADVANCED DOCUMENTATION

Please look at the source code (segueSelect/segueSelect.py) for an
overview of the advanced capabilities of this package. Some useful
functions are

    selectionFunction.check_consistency(plate)

which will calculate the KS probability that the spectropscopic sample
was drawn from the underlying photometric sample with the model
selection function.

    read_gdwarfs(file=_GDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=True,nocoords=False)

which reads the G stars (*not* just the dwarfs!) and applies the
color, SN, and E(B-V) cuts (same format as above). If galpy is
installed, velocities will also be transformed into the Galactic
coordinate frame (read the source for details).


##K STARS

The code can also determine the selection function for SEGUE K
stars. However, the bright/faint boundary seems to move around for K
stars as the survey progressed, so the default selection function
fails to give a reasonable selection function for many plates. The K
star selection function is still a work in progress, but determining
the bright/faint boundary on a plate-by-plate basis seems to work for
most plates. This can be done by using

    selectionFunction= segueSelect(sample='K',sn=15,select='all',indiv_brightlims=True)

Again, testing of this selection function has been very limited, so
use care when using the K stars.
