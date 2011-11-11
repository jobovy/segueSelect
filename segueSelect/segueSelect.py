import os, os.path
import sys
import copy
import math
import numpy
from scipy import special, interpolate, optimize, maxentropy, stats
import pyfits
import matplotlib
try:
    from galpy.util import bovy_plot
except ImportError:
    import bovy_plot
try:
    from galpy.util import bovy_coords
    _COORDSLOADED= True
except ImportError:
    _COORDSLOADED= False
########################SELECTION FUNCTION DETERMINATION#######################
_INTERPDEGREEBRIGHT= 3
_INTERPDEGREEFAINT= 3
_BINEDGES_G_FAINT= [0.,50.,70.,85.,200000000.]
_BINEDGES_G_BRIGHT= [0.,75.,150.,300.,200000000.]
###############################FILENAMES#######################################
_SEGUESELECTDIR=os.getenv('SEGUESELECTDIR')
_GDWARFALLFILE= os.path.join(_SEGUESELECTDIR,'gdwarfall_raw_nodups_ysl_nospec.fit')
_GDWARFFILE= os.path.join(_SEGUESELECTDIR,'gdwarf_raw_nodups_ysl_nospec.fit')
_KDWARFALLFILE= os.path.join(_SEGUESELECTDIR,'kdwarfall_raw_nodups_ysl_nospec.fit')
_KDWARFFILE= os.path.join(_SEGUESELECTDIR,'kdwarf_raw_nodups_ysl_nospec.fit')
_ERASESTR= "                                                                                "
class segueSelect:
    """Class that contains selection function for SEGUE targets"""
    def __init__(self,sample='G',plates=None,
                 select='all',
                 type_bright='tanhrcut',dr_bright=None,
                 interp_type_bright='tanh',
                 interp_degree_bright=_INTERPDEGREEBRIGHT,
                 robust_bright=True,
                 binedges_bright=_BINEDGES_G_BRIGHT,
                 type_faint='tanhrcut',dr_faint=None,
                 interp_type_faint='tanh',
                 interp_degree_faint=_INTERPDEGREEFAINT,
                 robust_faint=True,
                 binedges_faint=_BINEDGES_G_FAINT,
                 ug=False,ri=False,sn=True,
                 ebv=True,
                 _rmax=None,_rmin=None,indiv_brightlims=False,
                 _program_brightlims=False,
                 _platephot=None,_platespec=None,_spec=None):
        """
        NAME:
           __init__
        PURPOSE:
           load the selection function for this sample
        INPUT:
           sample= sample to load ('G', or 'K')
           select= 'all' selects all SEGUE stars in the color-range; 
                   'program' only selects program stars
           plates= if set, only consider this plate, or list of plates,
                   or 'faint'/'bright'plates only,
                   or plates '>1000' or '<2000'

           SELECTION FUNCTION DETERMINATION:
              default: tanhrcut for both bright and faint
           
              type_bright= type of selection function to determine 
                   'constant' for constant per plate; 
                   'r' universal function of r
                   'plateSN_r' function of r for plates in ranges in plateSN_r
                   'sharprcut' sharp cut in r for each plate, at the r-band mag of the faintest object on this plate
                   'tanhrcut' cut in r for each plate, at the r-band mag of the faintest object on this plate, with 0.1 mag tanh softening
              dr_bright= when determining the selection function as a function 
                         of r, binsize to use
              interp_degree_bright= when spline-interpolating, degree to use
              interp_type_bright= type of interpolation to use ('tanh' or 
                                   'spline')
              robust_bright= perform any fit robustly
              type_faint=, faint_dr, interp_degree_bright, interp_type_faint,
              robust_faint
              = same as the corresponding keywords for bright

              indiv_brightlims= if True, determine the bright/faint boundary as the brightest faint-plate spectrum, or the faintest bright-plate if there is no faint plate in the pair

           SPECTROSCOPIC SAMPLE SELECTION:
               ug= if True, cut on u-g, 
                     if list/array cut to ug[0] < u-g< ug[1]
               ri= if True, cut on r-i, 
                     if list/array cut to ri[0] < r-i< ri[1]
               sn= if False, don't cut on SN, 
                     if number cut on SN > the number (15)
               ebv= if True, cut on E(B-V), 
                    if number cut on EBV < the number (0.3)
        OUTPUT:
           object
        HISTORY:
           2011-07-08 - Written - Bovy@MPIA (NYU)
        """
        #Set options
        if dr_bright is None:
            if type_bright.lower() == 'r':
                dr_bright= 0.05
            elif type_bright.lower() == 'platesn_r':
                if sample.lower() == 'k':
                    dr_bright= 0.4
                elif sample.lower() == 'g':
                    dr_bright= 0.2
        if dr_faint is None:
            if type_faint.lower() == 'r':
                dr_faint= 0.2
            elif type_faint.lower() == 'platesn_r':
                if sample.lower() == 'g':
                    dr_faint= 0.2
                elif sample.lower() == 'k':
                    dr_faint= 0.5
        self.sample=sample.lower()
        #Load plates
        self.platestr= _load_fits(os.path.join(_SEGUESELECTDIR,
                                               'segueplates.fits'))
        #Add platesn_r to platestr
        platesn_r= (self.platestr.sn1_1+self.platestr.sn2_1)/2.
        self.platestr= _append_field_recarray(self.platestr,
                                              'platesn_r',platesn_r)
        if plates is None:
            self.plates= list(self.platestr.plate)
        else:
            if isinstance(plates,str):
                self.plates= self.platestr.plate
                if plates[0] == '>':
                    self.plates= self.plates[(self.plates > int(plates[1:len(plates)]))]
                elif plates[0] == '<':
                    self.plates= self.plates[(self.plates < int(plates[1:len(plates)]))]
                elif plates.lower() == 'faint':
                    indx= ['faint' in name for name in self.platestr.programname]
                    indx= numpy.array(indx,dtype='bool')
                    self.plates= self.plates[indx]
                elif plates.lower() == 'bright':
                    indx= [not 'faint' in name for name in self.platestr.programname]
                    indx= numpy.array(indx,dtype='bool')
                    self.plates= self.plates[indx]
                else:
                    print "'plates=' format not understood, check documentation"
                    return
                self.plates= list(self.plates)
            elif not isinstance(plates,(list,numpy.ndarray)):
                self.plates= [plates]
            elif isinstance(plates,numpy.ndarray):
                self.plates= list(plates)
            else:
                self.plates= plates
        #Remove 2820 for now BOVY DEAL WITH PLATE 2820, 2560, 2799, 2550
        if 2820 in self.plates:
            self.plates.remove(2820)
        if 2560 in self.plates:
            self.plates.remove(2560)
        if 2799 in self.plates:
            self.plates.remove(2799)
        if 2550 in self.plates:
            self.plates.remove(2550)
        #Remove duplicate plates
        self.plates= numpy.array(sorted(list(set(self.plates))))
        #Match platestr to plates again
        allIndx= numpy.arange(len(self.platestr),dtype='int')
        reIndx= numpy.zeros(len(self.plates),dtype='int')-1
        for ii in range(len(self.plates)):
            indx= (self.platestr.field('plate') == self.plates[ii])
            reIndx[ii]= (allIndx[indx][0])
        self.platestr= self.platestr[reIndx]
        #Build bright/faint dict
        self.platebright= {}
        for ii in range(len(self.plates)):
            p= self.plates[ii]
            if 'faint' in self.platestr[ii].programname:
                self.platebright[str(p)]= False
            else:
                self.platebright[str(p)]= True
        #Also build bright/faint index
        brightplateindx= numpy.empty(len(self.plates),dtype='bool') #BOVY: move this out of here
        faintplateindx= numpy.empty(len(self.plates),dtype='bool')
        for ii in range(len(self.plates)):
            if 'faint' in self.platestr[ii].programname: #faint plate
                faintplateindx[ii]= True
                brightplateindx[ii]= False 
            else:
                faintplateindx[ii]= False
                brightplateindx[ii]= True
        self.faintplateindx= faintplateindx
        self.brightplateindx= brightplateindx
        self.nbrightplates= numpy.sum(self.brightplateindx)
        self.nfaintplates= numpy.sum(self.faintplateindx)
        #Build plate-pair array
        platemate= numpy.zeros(len(self.plates),dtype='int')
        indices= numpy.arange(len(self.plates),dtype='int')
        for ii in range(len(self.plates)):
            plate= self.plates[ii]
            #Find plate's friend
            indx= (self.platestr.ra == self.platestr[ii].ra)
            if numpy.sum(indx) < 2: 
                platemate[ii]= -1 #No friend
                continue
            thisplates= self.plates[indx]
            jj= indices[indx][0]
            kk= indices[indx][1]
            if ii == kk: platemate[ii]= jj
            elif ii == jj: platemate[ii]= kk
        self.platemate= platemate
        #Set r limits
        if self.sample == 'g':
            self.rmin= 14.5
            self.rmax= 20.2
        elif self.sample == 'k':
            self.rmin= 14.5
            self.rmax= 19.
        if not _rmin is None: self.rmin= _rmin
        if not _rmax is None: self.rmax= _rmax
        #load the spectroscopic data
        self.select= select
        if _platespec is None:
            sys.stdout.write('\r'+"Reading and parsing spectroscopic data ...\r")
            sys.stdout.flush()
            if sample.lower() == 'g':
                if select.lower() == 'all':
                    self.spec= read_gdwarfs(ug=ug,ri=ri,sn=sn,
                                            ebv=ebv,nocoords=True)
                elif select.lower() == 'program':
                    self.spec= read_gdwarfs(file=_GDWARFFILE,
                                            ug=ug,ri=ri,sn=sn,
                                            ebv=ebv,nocoords=True)
            elif sample.lower() == 'k':
                if select.lower() == 'all':
                    self.spec= read_kdwarfs(ug=ug,ri=ri,sn=sn,
                                            ebv=ebv,nocoords=True)
                elif select.lower() == 'program':
                    self.spec= read_kdwarfs(file=_KDWARFFILE,
                                            ug=ug,ri=ri,sn=sn,
                                            ebv=ebv,nocoords=True)
            self.platespec= {}
            for plate in self.plates:
                #Find spectra for each plate
                indx= (self.spec.field('plate') == plate)
                self.platespec[str(plate)]= self.spec[indx]
            sys.stdout.write('\r'+_ERASESTR+'\r')
            sys.stdout.flush()
        else:
            self.platespec= _platespec
            self.spec= _spec
        #Set bright/faint divider
        if indiv_brightlims:
            if _program_brightlims and not select.lower() == 'program': #Grab the bright/faint interface from the program stars
                if sample.lower() == 'g':
                    bfspec= read_gdwarfs(file=_GDWARFFILE,
                                         ug=ug,ri=ri,sn=sn,
                                         ebv=ebv,nocoords=True)
                elif sample.lower() == 'k':
                    bfspec= read_kdwarfs(file=_KDWARFFILE,
                                            ug=ug,ri=ri,sn=sn,
                                            ebv=ebv,nocoords=True)
                bfplatespec= {}
                for plate in self.plates:
                    #Find spectra for each plate
                    indx= (bfspec.field('plate') == plate)
                    bfplatespec[str(plate)]= bfspec[indx]
            else:
                bfplatespec= self.platespec
            #Use brightest faint-plate object as the bright/faint interface
            faintbright= numpy.zeros(len(self.plates))
            for ii in range(len(self.plates)):
                #Pair?
                if not self.platemate[ii] == -1:
                    #Which one's faint?
                    if faintplateindx[ii]: #First one
                        if len(bfplatespec[str(self.plates[ii])].r) > 0:
                            faintbright[ii]= numpy.amin(bfplatespec[str(self.plates[ii])].r)
                        elif len(bfplatespec[str(self.plates[self.platemate[ii]])].r) > 0:
                            faintbright[ii]= numpy.amax(bfplatespec[str(self.plates[self.platemate[ii]])].r)
                        else: faintbright[ii]= 17.8
                    elif faintplateindx[self.platemate[ii]]: #Second one
                        if len(bfplatespec[str(self.plates[self.platemate[ii]])].r) > 0:
                            faintbright[ii]= numpy.amin(bfplatespec[str(self.plates[self.platemate[ii]])].r)
                        elif len(bfplatespec[str(self.plates[ii])].r) > 0:
                            faintbright[ii]= numpy.amax(bfplatespec[str(self.plates[ii])].r)
                        else:
                            faintbright[ii]= 17.8
                    else:
                        print "Error: no faint plate found for plate-pair %i,%i ..."%(self.plates[ii],self.plates[self.platemate[ii]])
                        print "Returning ..."
                        return None                        
                else:
                    if self.faintplateindx[ii]: #faint plate
                        faintbright[ii]= numpy.amin(bfplatespec[str(self.plates[ii])].r)
                    else:
                        faintbright[ii]= 17.8
                self.faintbright= faintbright
        else:
            self.faintbright= numpy.zeros(len(self.plates))+17.8
        #Also create faintbright dict
        self.faintbrightDict= {}
        for ii in range(len(self.plates)):
            self.faintbrightDict[str(self.plates[ii])]= self.faintbright[ii]
        #load the photometry for the SEGUE plates
        if _platephot is None:
            self.platephot= {}
            for ii in range(len(self.plates)):
                plate= self.plates[ii]
                sys.stdout.write('\r'+"Loading photometry for plate %i" % plate)
                sys.stdout.flush()
                platefile= os.path.join(_SEGUESELECTDIR,'segueplates',
                                        '%i.fit' % plate)
                self.platephot[str(plate)]= _load_fits(platefile)
                #Split into bright and faint
                if 'faint' in self.platestr[ii].programname:
                    indx= (self.platephot[str(plate)].field('r') >= self.faintbright[ii])
                    self.platephot[str(plate)]= self.platephot[str(plate)][indx]
                else:
                    indx= (self.platephot[str(plate)].field('r') < self.faintbright[ii])
                    self.platephot[str(plate)]= self.platephot[str(plate)][indx]
            sys.stdout.write('\r'+_ERASESTR+'\r')
            sys.stdout.flush()
        else:
            self.platephot= _platephot
        #Flesh out samples
        for plate in self.plates:
            if self.sample == 'g':
                indx= ((self.platephot[str(plate)].field('g')\
                            -self.platephot[str(plate)].field('r')) < 0.55)\
                            *((self.platephot[str(plate)].field('g')\
                                   -self.platephot[str(plate)].field('r')) > 0.48)\
                                   *(self.platephot[str(plate)].field('r') < 20.2)\
                                   *(self.platephot[str(plate)].field('r') > 14.5)
            elif self.sample == 'k':
                indx= ((self.platephot[str(plate)].field('g')\
                            -self.platephot[str(plate)].field('r')) > 0.55)\
                            *((self.platephot[str(plate)].field('g')\
                                   -self.platephot[str(plate)].field('r')) < 0.75)\
                                   *(self.platephot[str(plate)].field('r') < 19.)\
                                   *(self.platephot[str(plate)].field('r') > 14.5)
            self.platephot[str(plate)]= self.platephot[str(plate)][indx]
        #Determine selection function
        sys.stdout.write('\r'+"Determining selection function ...\r")
        sys.stdout.flush()
        if not numpy.sum(self.brightplateindx) == 0:
            self._determine_select(bright=True,type=type_bright,dr=dr_bright,
                                   interp_degree=interp_degree_bright,
                                   interp_type= interp_type_bright,
                                   robust=robust_bright,
                                   binedges=binedges_bright)
        if not numpy.sum(self.faintplateindx) == 0:
            self._determine_select(bright=False,type=type_faint,dr=dr_faint,
                                   interp_degree=interp_degree_faint,
                                   interp_type=interp_type_faint,
                                   robust=robust_faint,
                                   binedges=binedges_faint)
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        return None

    def __call__(self,plate,r=None,gr=None):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the selection function
        INPUT:
           plate - plate number
           r - dereddened r-band magnitude
           gr - dereddened g-r color
        OUTPUT:
           selection function
        HISTORY:
           2011-07-11 - Written - Bovy@MPIA (NYU)
        """
        #Handle input
        if isinstance(plate,(numpy.int16,int)) \
                and (isinstance(r,(int,float)) or r is None): #Scalar input
            plate= [plate]
            r= [r]
            scalarOut= True
        elif isinstance(plate,(numpy.int16,int)) \
                and isinstance(r,(list,numpy.ndarray)):
            #Special case this for optimization if sharprcut
            bright= self.platebright[str(plate)] #Short-cut
            if (bright and self.type_bright.lower() == 'sharprcut') \
                    or (not bright and self.type_faint.lower() == 'sharprcut'):
                nout= len(r)
                if isinstance(r,list): thisr= numpy.array(r)
                else: thisr= r
                out= numpy.zeros(nout)
                if bright:
                    indx= (thisr >= 14.5)*(thisr <= numpy.amin([self.rcuts[str(plate)],self.faintbrightDict[str(plate)]]))
                else:
                    indx= (thisr >= self.faintbrightDict[str(plate)])*(thisr <= numpy.amin([self.rcuts[str(plate)],self.rmax]))
                if numpy.sum(indx) == 0: return out
                out[indx]= self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]
                if isinstance(r,list): return list(out)
                else: return out
            elif (bright and self.type_bright.lower() == 'tanhrcut') \
                    or (not bright and self.type_faint.lower() == 'tanhrcut'):
                nout= len(r)
                if isinstance(r,list): thisr= numpy.array(r)
                else: thisr= r
                out= numpy.zeros(nout)
                if bright:
                    indx= (thisr >= 14.5)*(thisr <= self.faintbrightDict[str(plate)])
                else:
                    indx= (thisr >= self.faintbrightDict[str(plate)])*(thisr <= self.rmax)
                if numpy.sum(indx) == 0: return out
                out[indx]= self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]\
                    *_sf_tanh(thisr[indx],[self.rcuts[str(plate)]-0.1,
                                           -3.,0.])
                if isinstance(r,list): return list(out)
                else: return out
            else:
                if isinstance(r,numpy.ndarray):
                    plate= numpy.array([plate for ii in range(len(r))])
                else:
                    plate= [plate for ii in range(len(r))]
                scalarOut= False
        else:
            scalarOut= False
        out= []
        for ii in range(len(plate)):
            p= plate[ii]
            out.append(self._call_single(p,r[ii]))
        if isinstance(plate,numpy.ndarray):
            out= numpy.array(out)
        if scalarOut:
            return out[0]
        else:
            return out

    def _call_single(self,plate,r):
        """Call the selection function for a single object"""
        #First check whether this plate exists
        if not plate in self.plates: return 0.
        #First determine whether this is a bright or a faint plate
        bright= self.platebright[str(plate)] #Short-cut
        if bright:
            if not self.type_bright.lower() == 'tanhrcut+brightsharprcut' and (r >= self.faintbrightDict[str(plate)] or r < self.rmin): return 0.
            elif self.type_bright.lower() == 'constant':
                return self.weight[str(plate)]
            elif self.type_bright.lower() == 'r':
                if self.interp_type_bright.lower() == 'spline':
                    if r < self.s_one_r_bright_minxo:
                        return numpy.exp(_linear_func(r,
                                                      self.s_one_r_bright_minderiv,
                                                      self.s_one_r_bright_minxo,
                                                      self.s_one_r_bright_minyo))\
                                                      *self.weight[str(plate)]
                    else:
                        soner= numpy.exp(\
                            interpolate.splev(r,self.s_one_r_bright_interpolate))
                        if soner < 0.: return 0.
                        else: return self.weight[str(plate)]*soner
                elif self.interp_type_bright.lower() == 'tanh':
                    return _sf_tanh(r,self.s_one_r_tanh_params_bright)\
                        *self.weight[str(plate)] 
            elif self.type_bright.lower() == 'platesn_r':
                return self.platesn_sfs_bright[self.platesn_platebin_dict_bright[str(plate)]](plate,r=r)
            elif self.type_bright.lower() == 'sharprcut':
                if r <= self.rcuts[str(plate)]:
                    return self.weight[str(plate)]\
                        *self.rcuts_correct[str(plate)]
                else:
                    return 0.
            elif self.type_bright.lower() == 'tanhrcut':
                return self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]\
                    *_sf_tanh(r,[self.rcuts[str(plate)]-0.1,
                                 -3.,0.])
            elif self.type_bright.lower() == 'tanhrcut+brightsharprcut':
                if r <= self.rcuts_bright[str(plate)]: return 0.
                return self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]\
                    *_sf_tanh(r,[self.rcuts_faint[str(plate)]-0.1,
                                 -3.,0.])
        else:
            if not self.type_faint.lower() == 'tanhrcut+brightsharprcut' and (r < self.faintbrightDict[str(plate)] or r > self.rmax): return 0.
            elif self.type_faint.lower() == 'constant':
                return self.weight[str(plate)]
            elif self.type_faint.lower() == 'r':
                if self.interp_type_faint.lower() == 'spline':
                    if r < self.s_one_r_faint_minxo:
                        return numpy.exp(_linear_func(r,
                                                      self.s_one_r_faint_minderiv,
                                                      self.s_one_r_faint_minxo,
                                                      self.s_one_r_faint_minyo))\
                                                      *self.weight[str(plate)]
                    else:
                        soner= numpy.exp(\
                            interpolate.splev(r,self.s_one_r_faint_interpolate))
                        if soner < 0.: return 0.
                        else: return self.weight[str(plate)]*soner
                elif self.interp_type_faint.lower() == 'tanh':
                    return _sf_tanh(r,self.s_one_r_tanh_params_faint)\
                        *self.weight[str(plate)]
            elif self.type_faint.lower() == 'platesn_r':
                return self.platesn_sfs_faint[self.platesn_platebin_dict_faint[str(plate)]](plate,r=r)
            elif self.type_faint.lower() == 'sharprcut':
                if r <= self.rcuts[str(plate)]:
                    return self.weight[str(plate)]\
                        *self.rcuts_correct[str(plate)]
                else:
                    return 0.
            elif self.type_faint.lower() == 'tanhrcut':
                return self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]\
                    *_sf_tanh(r,[self.rcuts[str(plate)]-0.1,
                                 -3.,0.])
            elif self.type_faint.lower() == 'tanhrcut+brightsharprcut':
                if r <= self.rcuts_bright[str(plate)]: return 0.
                return self.weight[str(plate)]\
                    *self.rcuts_correct[str(plate)]\
                    *_sf_tanh(r,[self.rcuts_faint[str(plate)]-0.1,
                                 -3.,0.])

    def check_consistency(self,plate):
        """
        NAME:
           check_consistency
        PURPOSE:
           calculate the KS probability that this plate is consistent with 
           being drawn from the underlying photometrix sample using our model
           for the selection function
        INPUT:
           pate - plate number(s), 'all', 'bright', or 'faint'
        OUTPUT:
           KS probability or list/array of such numbers
        HISTORY:
           2011-07-21 - Written - Bovy@MPIA (NYU)
        """
        #Handle input
        scalarOut= False
        if isinstance(plate,str) and plate.lower() == 'all':
            plate= self.plates
        elif isinstance(plate,str) and plate.lower() == 'bright':
            plate= self.plates[self.brightplateindx]
        elif isinstance(plate,str) and plate.lower() == 'faint':
            plate= self.plates[self.faintplateindx]
        if isinstance(plate,(numpy.int16,int)): #Scalar input
            plate= [plate]
            scalarOut= True
        out= []
        for p in plate:
            out.append(self._check_consistency_single(p))
        if scalarOut: return out[0]
        elif isinstance(plate,numpy.ndarray): return numpy.array(out)
        else: return out

    def _check_consistency_single(self,plate):
        """check_consistency for a single plate"""
        photr,specr,fn1,fn2= self._plate_rcdfs(plate)
        if photr is None:
            return -1
        j1, j2, i= 0, 0, 0
        id1= range(len(photr)+len(specr))
        id2= range(len(photr)+len(specr))
        while j1 < len(photr) and j2 < len(specr):
            d1= photr[j1]
            d2= specr[j2]
            if d1 <= d2: j1+= 1
            if d2 <= d1: j2+= 1
            id1[i]= j1
            id2[i]= j2
            i+= 1
        id1= id1[0:i-1]
        id2= id2[0:i-1]
        D= numpy.amax(numpy.fabs(fn1[id1]-fn2[id2]))
        neff= len(photr)*len(specr)/float(len(photr)+len(specr))
        return stats.ksone.sf(D,neff)

    def _plate_rcdfs(self,plate):
        #Load photometry and spectroscopy for this plate
        thisplatephot= self.platephot[str(plate)]
        thisplatespec= self.platespec[str(plate)]
        #Cut to bright or faint part
        if self.platebright[str(plate)]:
            thisplatespec= thisplatespec[(thisplatespec.dered_r < self.faintbrightDict[str(plate)])\
                                             *(thisplatespec.dered_r > self.rmin)]
        else:
            thisplatespec= thisplatespec[(thisplatespec.dered_r < self.rmax)\
                                             *(thisplatespec.dered_r >= self.faintbrightDict[str(plate)])]
        if len(thisplatespec.dered_r) == 0: return (None,None,None,None)
        #Calculate selection function weights for the photometry
        w= numpy.zeros(len(thisplatephot.r))
        for ii in range(len(w)):
            w[ii]= self(plate,r=thisplatephot[ii].r)
        #Calculate KS test statistic
        sortindx_phot= numpy.argsort(thisplatephot.r)
        sortindx_spec= numpy.argsort(thisplatespec.dered_r)
        sortphot= thisplatephot[sortindx_phot]
        sortspec= thisplatespec[sortindx_spec]
        w= w[sortindx_phot]
        fn1= numpy.cumsum(w)/numpy.sum(w)
        fn2= numpy.ones(len(sortindx_spec))
        fn2= numpy.cumsum(fn2)
        fn2/= fn2[-1]
        return (sortphot.r,sortspec.dered_r,fn1,fn2)

    def plot_plate_rcdf(self,plate,overplot=False,xrange=None,yrange=None,
                        photcolor='k',speccolor='r'):
        """
        NAME:
           plot_plate_rcdf
        PURPOSE:
           plot the r-band magnitude CDF for the photometric sample * selection
           function model and for the spectroscopic sample for a single plate
        INPUT:
           plate - plate to plot
           overplot= of True, overplot
           xrange=, yrange=
           photcolor=, speccolor= color to use
        OUTPUT:
           plot
        HISTORY:
           2011-07-21 - Written - Bovy@MPIA (NYU)
        """
        photr,specr,fn1,fn2= self._plate_rcdfs(plate)
        if photr is None:
            print "Plate %i has no spectroscopic data ..." % plate
            print "Returning ..."
            return None           
        if xrange is None: xrange= [nu.amin([nu.amin(photr),nu.amin(specr)])-0.1,
                                    nu.amax([nu.amax(photr),nu.amax(specr)])+0.1]
        if yrange is None: yrange= [0.,1.1]
        bovy_plot.bovy_plot(photr,fn1,photcolor+'-',overplot=overplot)
        bovy_plot.bovy_plot(specr,fn2,speccolor+'-',overplot=True)
        return None

    def plot(self,x='r',y='sf',plate='a bright plate',overplot=False):
        """
        NAME:
           plot
        PURPOSE:
           plot the derived selection function
        INPUT:
           x= what to plot on the x-axis (e.g., 'r')
           y= what to plot on the y-axis (default function value)
           plate= plate to plot (number or 'a bright plate' (default), 'a faint plate')
           overplot= if True, overplot
        OUTPUT:
           plot to output
        HISTORY:
           2011-07-18 - Written - Bovy@MPIA (NYU)
        """
        _NXS= 1001
        if isinstance(plate,str) and plate.lower() == 'a bright plate':
            plate= 2964
        elif isinstance(plate,str) and plate.lower() == 'a faint plate':
            plate= 2965
        if x.lower() == 'r':
            xs= numpy.linspace(self.rmin,self.rmax,_NXS)
            xrange= [self.rmin,self.rmax]
            xlabel= r'$r_0\ [\mathrm{mag}]$'
        #Evaluate selection function
        zs= self(plate,r=xs)
        if y.lower() == 'sf':
            ys= zs
            ylabel= r'$\mathrm{selection\ function}$'
            yrange= [0.,1.2*numpy.amax(ys)]
        bovy_plot.bovy_plot(xs,ys,'k-',xrange=xrange,yrange=yrange,
                            xlabel=xlabel,ylabel=ylabel,
                            overplot=overplot)
        #Also plot data
        if self.type.lower() == 'r':
            pindx= (self.plates == plate)
            if self.platebright[str(plate)]:
                bovy_plot.bovy_plot(self.s_r_plate_rs_bright,
                                    self.s_r_plate_bright[:,pindx],
                                    color='k',
                                    marker='o',ls='none',overplot=True)
                
            else:
                bovy_plot.bovy_plot(self.s_r_plate_rs_faint,
                                    self.s_r_plate_faint[:,pindx],
                                    color='k',
                                    marker='o',ls='none',overplot=True)
        return None

    def plot_s_one_r(self,plate='a bright plate',overplot=False,color='k',
                     xrange=None,yrange=None):
        """
        NAME:
           plot_s_one_r
        PURPOSE:
           plot the derived selection function s_1(r)
        INPUT:
           plate= plate to plot (number or 'a bright plate' (default), 
                                 'a faint plate')
           overplot= if True, overplot
           xrange=, yrange=
        OUTPUT:
           plot to output
        HISTORY:
           2011-07-20 - Written - Bovy@MPIA (NYU)
        """
        _NXS= 1001
        if isinstance(plate,str) and plate.lower() == 'a bright plate':
            plate= 2964
        elif isinstance(plate,str) and plate.lower() == 'a faint plate':
            plate= 2965
        xs= numpy.linspace(self.rmin+0.001,self.rmax-0.001,_NXS)
        if xrange is None: xrange= [self.rmin,self.rmax]
        xlabel= r'$r\ [\mathrm{mag}]$'
        #Evaluate selection function
        ys= numpy.array(self(plate,r=xs))/self.weight[str(plate)]
        ylabel= r'$r\ \mathrm{dependence\ of\ selection\ function}$'
        if yrange is None: yrange= [0.,1.2*numpy.amax(ys)]
        bovy_plot.bovy_plot(xs,ys,color+'-',xrange=xrange,yrange=yrange,
                            xlabel=xlabel,ylabel=ylabel,
                            overplot=overplot)
        pindx= (self.plates == plate)
        if (self.brightplateindx[pindx][0] \
                and self.type_bright.lower() != 'r')\
                or (self.faintplateindx[pindx][0] \
                        and self.type_faint.lower() != 'r'): return
        #Also plot data
        from matplotlib.pyplot import errorbar
        if self.platebright[str(plate)]:
            bovy_plot.bovy_plot(self.s_r_plate_rs_bright,
                                self.s_one_r_bright,
                                color=color,
                                marker='o',ls='none',overplot=True)
            errorbar(self.s_r_plate_rs_bright,
                     self.s_one_r_bright,
                     self.s_one_r_err_bright,
                     xerr= numpy.zeros(len(self.interp_rs_bright))+(self.interp_rs_bright[1]-self.interp_rs_bright[0])/2.,
                     fmt=None,ecolor=color)
        else:
            bovy_plot.bovy_plot(self.s_r_plate_rs_faint,
                                self.s_one_r_faint,
                                color=color,
                                marker='o',ls='none',overplot=True)
            errorbar(self.s_r_plate_rs_faint,
                     self.s_one_r_faint,
                     self.s_one_r_err_faint,
                     xerr= numpy.zeros(len(self.interp_rs_faint))+(self.interp_rs_faint[1]-self.interp_rs_faint[0])/2.,
                     fmt=None,ecolor=color)
        return None

    def plotColorMag(self,x='gr',y='r',plate='all',spec=False,scatterplot=True,
                     bins=None,specbins=None):
        """
        NAME:
           plotColorMag
        PURPOSE:
           plot the distribution of photometric/spectroscopic objects in color
           magnitude (or color-color) space
        INPUT:
           x= what to plot on the x-axis (combinations of ugriz as 'g', 
               or 'gr')
           y= what to plot on the y-axis (combinations of ugriz as 'g',  
               or 'gr')
           plate= plate(s) to plot, int or list/array, 'all', 'bright', 'faint'
           spec= if True, overlay spectroscopic objects as red contours and 
                 histograms
           scatterplot= if False, regular scatterplot, 
                        if True, hogg_scatterplot
           bins= number of bins to use in the histogram(s)
           specbins= number of bins to use in histograms of spectropscopic 
                     objects
       OUTPUT:
        HISTORY:
           2011-07-13 - Written - Bovy@MPIA (NYU)
        """
        if isinstance(plate,str) and plate.lower() == 'all':
            plate= self.plates
        elif isinstance(plate,str) and plate.lower() == 'bright':
            plate= []
            for ii in range(len(self.plates)):
                if not 'faint' in self.platestr[ii].programname:
                    plate.append(self.plates[ii])
        elif isinstance(plate,str) and plate.lower() == 'faint':
            plate= []
            for ii in range(len(self.plates)):
                if 'faint' in self.platestr[ii].programname:
                    plate.append(self.plates[ii])
        elif isinstance(plate,(list,numpy.ndarray)):
            plate=plate
        else:
            plate= [plate]
        xs, ys= [], []
        specxs, specys= [], []
        for ii in range(len(plate)):
            p=plate[ii]
            thisplatephot= self.platephot[str(p)]
            thisplatespec= self.platespec[str(p)]
            if len(x) > 1: #Color
                xs.extend(thisplatephot.field(x[0])\
                              -thisplatephot.field(x[1])) #dereddened
                specxs.extend(thisplatespec.field('dered_'+x[0])\
                                  -thisplatespec.field('dered_'+x[1]))
            else:
                xs.extend(thisplatephot.field(x[0]))
                specxs.extend(thisplatespec.field('dered_'+x[0]))
            if len(y) > 1: #Color
                ys.extend(thisplatephot.field(y[0])\
                              -thisplatephot.field(y[1])) #dereddened
                specys.extend(thisplatespec.field('dered_'+y[0])\
                                  -thisplatespec.field('dered_'+y[1]))
            else:
                ys.extend(thisplatephot.field(y[0]))
                specys.extend(thisplatespec.field('dered_'+y[0]))
        xs= numpy.array(xs)
        xs= numpy.reshape(xs,numpy.prod(xs.shape))
        ys= numpy.array(ys)
        ys= numpy.reshape(ys,numpy.prod(ys.shape))
        specxs= numpy.array(specxs)
        specxs= numpy.reshape(specxs,numpy.prod(specxs.shape))
        specys= numpy.array(specys)
        specys= numpy.reshape(specys,numpy.prod(specys.shape))
        if len(x) > 1:
            xlabel= '('+x[0]+'-'+x[1]+')_0'
        else:
            xlabel= x[0]+'_0'
        xlabel= r'$'+xlabel+r'$'
        if len(y) > 1:
            ylabel= '('+y[0]+'-'+y[1]+')_0'
        else:
            ylabel= y[0]+'_0'
        ylabel= r'$'+ylabel+r'$'
        if len(x) > 1: #color
            xrange= [numpy.amin(xs)-0.02,numpy.amax(xs)+0.02]
        else:
            xrange= [numpy.amin(xs)-0.7,numpy.amax(xs)+0.7]
        if len(y) > 1: #color
            yrange= [numpy.amin(ys)-0.02,numpy.amax(ys)+0.02]
        else:
            yrange= [numpy.amin(ys)-0.7,numpy.amax(ys)+0.7]
        if bins is None:
            bins= int(numpy.ceil(0.3*numpy.sqrt(len(xs))))
        if specbins is None: specbins= bins
        if scatterplot:
            if len(xs) > 100000: symb= 'w,'
            else: symb= 'k,'
            if spec:
                #First plot spectroscopic sample
                cdict = {'red': ((.0, 1.0, 1.0),
                                 (1.0, 1.0, 1.0)),
                         'green': ((.0, 1.0, 1.0),
                                   (1.0, 1.0, 1.0)),
                         'blue': ((.0, 1.0, 1.0),
                                  (1.0, 1.0, 1.0))}
                allwhite = matplotlib.colors.LinearSegmentedColormap('allwhite',cdict,256)
                speclevels= list(special.erf(0.5*numpy.arange(1,4)))
                speclevels.append(1.01)#HACK TO REMOVE OUTLIERS
                bovy_plot.scatterplot(specxs,specys,symb,onedhists=True,
                                      levels=speclevels,
                                      onedhistec='k',
                                      cntrcolors='w',
                                      onedhistls='dashed',
                                      onedhistlw=1.5,
                                      cmap=allwhite,
                                      xlabel=xlabel,ylabel=ylabel,
                                      xrange=xrange,yrange=yrange,
                                      bins=specbins)
                
            bovy_plot.scatterplot(xs,ys,symb,onedhists=True,
                                  xlabel=xlabel,ylabel=ylabel,
                                  xrange=xrange,yrange=yrange,bins=bins,
                                  overplot=spec)
        else:
            bovy_plot.bovy_plot(xs,ys,'k,',onedhists=True,
                                xlabel=xlabel,ylabel=ylabel,
                                xrange=xrange,yrange=yrange)
        return None                

    def _determine_select(self,bright=True,type=None,dr=None,
                          interp_degree=_INTERPDEGREEBRIGHT,
                          interp_type='tanh',
                          robust=False,
                          binedges=None):
        """Function that actually determines the selection function"""
        if bright:
            self.type_bright= type
            plateindx= self.brightplateindx
        else:
            self.type_faint= type
            plateindx= self.faintplateindx
        if type.lower() == 'platesn_r': #plateSN_r dependent r selection
            #Divide up plates in bins
            nbins= len(binedges)-1
            plate_in_bins= [[] for ii in range(nbins)]
            platebin_dict= {}
            theseplates= self.plates[plateindx]
            thisplatestr= self.platestr[plateindx]
            for ii in range(len(theseplates)):
                kk= 0
                while kk < nbins \
                        and thisplatestr[ii].platesn_r > binedges[kk+1]:
                    kk+=1
                plate_in_bins[kk].append(theseplates[ii])
                #Also create dictionary with bin for each plate
                platebin_dict[str(theseplates[ii])]= kk              
            #For each set of plates, instantiate new selection object
            platesn_sfs= []
            for kk in range(nbins):
                if bright:
                    type_faint= 'constant'
                    type_bright= 'r'
                else:
                    type_faint= 'r'
                    type_bright= 'constant'
                platesn_sfs.append(segueSelect(sample=self.sample,
                                               plates=plate_in_bins[kk],
                                               select=self.select,
                                               type_bright=type_bright,
                                               dr_bright=dr,
                                               interp_type_bright='tanh',
                                               interp_degree_bright=interp_degree,
                                               robust_bright=robust,
                                               type_faint=type_faint,
                                               dr_faint=dr,
                                               interp_type_faint='tanh',
                                               interp_degree_faint=interp_degree,
                                               robust_faint=robust,
                                               _platephot=copy.copy(self.platephot),
                                               _platespec=copy.copy(self.platespec)
                                               ,_spec=copy.copy(self.spec)))
            if bright:
                self.platesn_plate_in_bins_bright= plate_in_bins
                self.platesn_platebin_dict_bright= platebin_dict
                self.platesn_sfs_bright= platesn_sfs
            else:
                self.platesn_plate_in_bins_faint= plate_in_bins
                self.platesn_sfs_faint= platesn_sfs
                self.platesn_platebin_dict_faint= platebin_dict
            return None #Done here!
        #First determine the total weight for each plate
        if not hasattr(self,'weight'): self.weight= {}
        for ii in range(len(self.plates)):
            if bright and 'faint' in self.platestr[ii].programname: continue
            elif not bright \
                    and not 'faint' in self.platestr[ii].programname: continue
            plate= self.plates[ii]
            self.weight[str(plate)]= len(self.platespec[str(plate)])\
                /float(len(self.platephot[str(plate)]))
        if type.lower() == 'constant':
            return #We're done!
        if type.lower() == 'sharprcut' or type.lower() == 'tanhrcut':
            #For each plate cut at the location of the faintest object
            if not hasattr(self,'rcuts'): self.rcuts= {}
            if not hasattr(self,'rcuts_correct'): self.rcuts_correct= {}
            for ii in range(len(self.plates)):
                if bright and 'faint' in self.platestr[ii].programname: continue
                elif not bright \
                        and not 'faint' in self.platestr[ii].programname: continue
                p= self.plates[ii]
                if self.weight[str(p)] == 0.:
                    self.rcuts[str(p)]= 0.
                    self.rcuts_correct[str(p)]= 0.
                    continue
                self.rcuts[str(p)]= numpy.amax(self.platespec[str(p)].dered_r)
                denom= float(numpy.sum((self.platephot[str(p)].r <= self.rcuts[str(p)])))
                if denom == 0.: self.rcuts_correct[str(p)]= 0.
                else:
                    self.rcuts_correct[str(p)]= \
                        float(len(self.platephot[str(p)]))/denom
        elif type.lower() == 'tanhrcut+brightsharprcut':
            #For each plate cut at the location of the brightest and faintest object
            if not hasattr(self,'rcuts_faint'): self.rcuts_faint= {}
            if not hasattr(self,'rcuts_bright'): self.rcuts_bright= {}
            if not hasattr(self,'rcuts_correct'): self.rcuts_correct= {}
            for ii in range(len(self.plates)):
                if bright and 'faint' in self.platestr[ii].programname: continue
                elif not bright \
                        and not 'faint' in self.platestr[ii].programname: continue
                p= self.plates[ii]
                if self.weight[str(p)] == 0.:
                    self.rcuts_bright[str(p)]= 0.
                    self.rcuts_faint[str(p)]= 0.
                    self.rcuts_correct[str(p)]= 0.
                    continue
                self.rcuts_bright[str(p)]= numpy.amin(self.platespec[str(p)].dered_r)
                self.rcuts_faint[str(p)]= numpy.amax(self.platespec[str(p)].dered_r)
                denom= float(numpy.sum((self.platephot[str(p)].r <= self.rcuts_faint[str(p)])*(self.platephot[str(p)].r > self.rcuts_bright[str(p)])))
                if denom == 0.: self.rcuts_correct[str(p)]= 0.
                else:
                    self.rcuts_correct[str(p)]= \
                        float(len(self.platephot[str(p)]))/denom
        elif type.lower() == 'r':
            #Determine the selection function in bins in r, for bright/faint
            nrbins= int(math.floor((17.8-self.rmin)/dr))+1
            s_one_r= numpy.zeros((nrbins,len(self.plates)))
            s_r= numpy.zeros((nrbins,len(self.plates)))
            #Determine s_1(r) for each plate separately first
            weights= numpy.zeros(len(self.plates))
            if not bright:
                thisrmin, thisrmax= 17.8, self.rmax+dr/2. #slightly further to avoid out-of-range errors
            else:
                thisrmin, thisrmax= self.rmin-dr/2., 17.8 #slightly further to avoid out-of-range errors
            for ii in range(len(self.plates)):
                plate= self.plates[ii]
                if bright and 'faint' in self.platestr[ii].programname: 
                    continue
                elif not bright \
                        and not 'faint' in self.platestr[ii].programname: 
                    continue
                nspecr, edges = numpy.histogram(self.platespec[str(plate)].dered_r,bins=nrbins,range=[thisrmin,thisrmax])
                nphotr, edges = numpy.histogram(self.platephot[str(plate)].r,
                                                bins=nrbins,
                                                range=[thisrmin,thisrmax])
                nspecr= numpy.array(nspecr,dtype='float64')
                nphotr= numpy.array(nphotr,dtype='float64')
                nonzero= (nspecr > 0.)*(nphotr > 0.)
                s_r[nonzero,ii]= nspecr[nonzero].astype('float64')/nphotr[nonzero]
                weights[ii]= float(numpy.sum(nspecr))/float(numpy.sum(nphotr))
                nspecr/= float(numpy.sum(nspecr))
                nphotr/= float(numpy.sum(nphotr))
                s_one_r[nonzero,ii]= nspecr[nonzero]/nphotr[nonzero]
            if bright:
                self.s_r_plate_rs_bright= \
                    numpy.linspace(self.rmin+dr/2.,17.8-dr/2.,nrbins)
                self.s_r_plate_bright= s_r
                self.s_one_r_plate_bright= s_one_r
            else:
                self.s_r_plate_rs_faint= \
                    numpy.linspace(17.8+dr/2.,self.rmax-dr/2.,nrbins)
                self.s_r_plate_faint= s_r
                self.s_one_r_plate_faint= s_one_r
            s_one_r_plate= s_one_r
            s_r_plate= s_r
            fromIndividual= False
            if fromIndividual:
                #Mean or median?
                median= False
                if median:
                    s_one_r= numpy.median(s_one_r_plate[:,plateindx],axis=1)
                else:
                    if bright:
                        s_one_r= numpy.sum(s_one_r_plate,axis=1)/self.nbrightplates
                    else:
                        s_one_r= numpy.sum(s_one_r_plate,axis=1)/self.nfaintplates
            else:
                s_one_r= \
                    numpy.sum(s_r_plate[:,plateindx],axis=1)\
                    /numpy.sum(weights)
            if bright:
                self.s_one_r_bright= s_one_r
                self.s_r_bright= s_r
            else:
                self.s_one_r_faint= s_one_r
                self.s_r_faint= s_r
            #Bootstrap an uncertainty on the selection function
            if bright: nplates= self.nbrightplates
            else: nplates= self.nfaintplates
            jack_samples= numpy.zeros((nplates,len(s_one_r)))
            jack_s_r_plate= s_r_plate[:,plateindx]
            jack_s_r_weights= weights[plateindx]
            for jj in range(nplates):
                boot_indx= numpy.array([True for ii in range(nplates)],\
                                           dtype='bool')
                boot_indx[jj]= False
                if fromIndividual:
                    #Mean or median?
                    if median:
                        jack_samples[jj,:]= numpy.median(s_one_r_plate[:,plateindx[boot_indx]],
                                              axis=1)
                    else:
                        jack_samples[jj,:]= numpy.sum(s_one_r_plate[:,plateindx[boot_indx]],
                                           axis=1)/nplates
                else:
                    jack_samples[jj,:]= \
                        numpy.sum(jack_s_r_plate[:,boot_indx],axis=1)\
                        /numpy.sum(jack_s_r_weights[boot_indx])
            #Compute jackknife uncertainties
            s_one_r_err= numpy.sqrt((nplates-1)*numpy.var(jack_samples,
                                                                 axis=0))
            s_one_r_err[(s_one_r_err == 0.)]= 0.01
            if bright:
                self.s_one_r_jack_samples_bright= jack_samples
                self.s_one_r_err_bright= s_one_r_err
            else:
                self.s_one_r_jack_samples_faint= jack_samples
                self.s_one_r_err_faint= s_one_r_err
            if bright: self.interp_type_bright= interp_type
            else: self.interp_type_faint= interp_type
            if bright:
                w= numpy.zeros(len(self.s_one_r_bright))+10000.
                yfunc= numpy.zeros(len(w))-20.
                nonzero= (self.s_one_r_bright > 0.)
                w[nonzero]= \
                    self.s_one_r_bright[nonzero]/self.s_one_r_err_bright[nonzero]
                yfunc[nonzero]= numpy.log(self.s_one_r_bright[nonzero])
                self.interp_rs_bright= \
                    numpy.linspace(self.rmin+1.*dr/2.,17.8-1.*dr/2.,nrbins)
                if interp_type.lower() == 'spline':
                    self.s_one_r_bright_interpolate= interpolate.splrep(\
                        self.interp_rs_bright,yfunc,
                        k=interp_degree,w=w)
                    #Continue along the derivative for out of bounds
                    minderiv= interpolate.splev(self.interp_rs_bright[0],
                                                self.s_one_r_bright_interpolate,
                                                der=1)
                    self.s_one_r_bright_minderiv= minderiv
                    self.s_one_r_bright_minxo= self.interp_rs_bright[0]
                    self.s_one_r_bright_minyo= yfunc[0]
                elif interp_type.lower() == 'tanh':
                    #Fit a tanh to s_1(r)
                    params= numpy.array([17.7,numpy.log(0.1),
                                         numpy.log(3.)])
                    params= optimize.fmin_powell(_sf_tanh_minusloglike,
                                                 params,
                                                 args=(self.interp_rs_bright,
                                                       self.s_one_r_bright,
                                                       self.s_one_r_err_bright,
                                      numpy.zeros(len(self.interp_rs_bright))+(self.interp_rs_bright[1]-self.interp_rs_bright[0])/2.,
                                                       robust))
                    self.s_one_r_tanh_params_bright= params
            else:
                w= numpy.zeros(len(self.s_one_r_faint))+10000.
                yfunc= numpy.zeros(len(w))-20.
                nonzero= (self.s_one_r_faint > 0.)
                w[nonzero]= \
                    self.s_one_r_faint[nonzero]/self.s_one_r_err_faint[nonzero]
                yfunc[nonzero]= numpy.log(self.s_one_r_faint[nonzero])
                self.interp_rs_faint= \
                    numpy.linspace(17.8+1.*dr/2.,self.rmax-dr/2.,nrbins)
                if interp_type.lower() == 'spline':
                    self.s_one_r_faint_interpolate= interpolate.splrep(\
                        self.interp_rs_faint,yfunc,
                        k=interp_degree,w=w)
                    #Continue along the derivative for out of bounds
                    minderiv= interpolate.splev(self.interp_rs_faint[0],
                                                self.s_one_r_faint_interpolate,
                                                der=1)
                    self.s_one_r_faint_minderiv= minderiv
                    self.s_one_r_faint_minxo= self.interp_rs_faint[0]
                    self.s_one_r_faint_minyo= yfunc[0]
                elif interp_type.lower() == 'tanh':
                    #Fit a tanh to s_1(r)
                    params= numpy.array([18.7,numpy.log(0.1),
                                         numpy.log(3.)])
                    params= optimize.fmin_powell(_sf_tanh_minusloglike,
                                                 params,
                                                 args=(self.interp_rs_faint,
                                                       self.s_one_r_faint,
                                                       self.s_one_r_err_faint,
                                                       numpy.zeros(len(self.interp_rs_faint))+(self.interp_rs_faint[1]-self.interp_rs_faint[0])/2.,robust))
                    self.s_one_r_tanh_params_faint= params
            return None

def _sf_tanh(r,params):
    """Tanh description of the selection,
    params=[rcentral,logsigmar,logconstant]"""
    return math.exp(params[2])/2.*(1.-numpy.tanh((r-params[0])/math.exp(params[1])))

def _sf_tanh_minusloglike(params,rs,sfs,sferrs,rerrs=None,robust=False):
    #return 0.5*numpy.sum((sfs-_sf_tanh(rs,params))**2./2./sferrs**2.)
    #Robust
    if rerrs is None:
        if robust:
            return numpy.sum(numpy.fabs((sfs-_sf_tanh(rs,params))/sferrs))
        else:
            return numpy.sum((sfs-_sf_tanh(rs,params))**2./2./sferrs**2.)
    else:
        ngrid= 21
        nsigma= 3.
        grid= numpy.linspace(-nsigma,nsigma,ngrid)
        if robust:
            presum= numpy.fabs(grid)
        else:
            presum= grid**2./2.
        out= 0.
        for ii in range(len(rs)):
            thisgrid= grid*rerrs[ii]+rs[ii]
            if robust:
                out+= maxentropy.logsumexp(presum+numpy.fabs(sfs[ii]-_sf_tanh(thisgrid,
                                                                              params))/\
                                               sferrs[ii])
            else:
                out+= maxentropy.logsumexp(presum+(sfs[ii]-_sf_tanh(thisgrid,
                                                                    params))**2./2./\
                                               sferrs[ii]**2.)
        return out
            

def _linear_func(x,deriv,xo,yo):
    """Evaluate a linear function"""
    return deriv*(x-xo)+yo

def ivezic_dist_gr(g,r,feh,dg=0.,dr=0.,dfeh=0.,return_error=False,
                   dmr=0.1):
    """
    NAME:
       ivezic_dist_gr
    PURPOSE:
        Ivezic et al. (2008) distances in terms of g-r for <M0 stars
    INPUT:
       g, r, feh - dereddened g and r and metallicity
       return_error= if True, return errors
       dg, dr, dfeh= uncertainties
       dmr= intrinsic cmd scatter
    OUTPUT:
       (dist,disterr) arrays in kpc
    HISTORY:
       2011-07-11 - Written - Bovy@MPIA (NYU)
    """
    #First distances, then uncertainties
    gi= _gi_gr(g-r)
    mr= _mr_gi(gi,feh)
    ds= 10.**(0.2*(r-mr)-2.)
    if not return_error: return (ds,0.*ds)
    #Now propagate the uncertainties
    dgi= numpy.sqrt(_gi_gr(g-r,dg=True)**2.*dg**2.
                    +_gi_gr(g-r,dr=True)**2.*dr**2.)
    dmr= numpy.sqrt(_mr_gi(gi,feh,dgi=True)**2.*dgi**2.
                    +_mr_gi(gi,feh,dfeh=True)**2.*dfeh**2.+dmr**2.)
    derrs= 0.2*numpy.log(10.)*numpy.sqrt(dmr**2.+dr**2.)*ds
    return (ds,derrs)

def juric_dist_gr(g,r,dg=0.,dr=0.,return_error=False,
                  dmr=0.3,faint=False):
    """
    NAME:
       juric_dist_gr
    PURPOSE:
        Juric et al. (2008) distances in terms of g-r for <M0 stars
    INPUT:
       g, r- dereddened g and r
       return_error= if True, return errors
       dg, dr= uncertainties
       dmr= intrinsic cmd scatter
       faint= if True, use faint relation, else use bright
    OUTPUT:
       (dist,disterr) arrays in kpc
    HISTORY:
       2011-08-08 - Written - Bovy (NYU)
    """
    #First distances, then uncertainties
    ri= _ri_gr(g-r)
    if faint:
        mr= _mr_ri_faint(ri)
    else:
        mr= _mr_ri_bright(ri)
    ds= 10.**(0.2*(r-mr)-2.)
    if not return_error: return (ds,0.*ds)
    #Now propagate the uncertainties
    dri= numpy.sqrt(_ri_gr(g-r,dg=True)**2.*dg**2.
                    +_ri_gr(g-r,dr=True)**2.*dr**2.)
    if faint:
        dmr= numpy.sqrt(_mr_ri_faint(ri,dri=True)**2.*dri**2.
                        +dmr**2.)
    else:
        dmr= numpy.sqrt(_mr_ri_bright(ri,dri=True)**2.*dri**2.
                        +dmr**2.)
    derrs= 0.2*numpy.log(10.)*numpy.sqrt(dmr**2.+dr**2.)*ds
    return (ds,derrs)

def read_gdwarfs(file=_GDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=True,nocoords=False):
    """
    NAME:
       read_gdwarfs
    PURPOSE:
       read the spectroscopic G dwarf sample
    INPUT:
       logg= if True, cut on logg, if number, cut on logg > the number (3.75)
       ug= if True, cut on u-g, if list/array cut to ug[0] < u-g< ug[1]
       ri= if True, cut on r-i, if list/array cut to ri[0] < r-i< ri[1]
       sn= if False, don't cut on SN, if number cut on SN > the number (15)
       ebv= if True, cut on E(B-V), if number cut on EBV < the number (0.3)
       nocoords= if True, don't calculate distances or transform coordinates
    OUTPUT:
       cut data, returns numpy.recarray
    HISTORY:
       2011-07-08 - Written - Bovy@MPIA (NYU)
    """
    raw= _load_fits(file)
    #First cut on r
    indx= (raw.field('dered_r') < 20.2)*(raw.field('dered_r') > 14.5)
    raw= raw[indx]
    #Then cut on g-r
    indx= ((raw.field('dered_g')-raw.field('dered_r')) < 0.55)\
        *((raw.field('dered_g')-raw.field('dered_r')) > .48)
    raw= raw[indx]
    #Cut on velocity errs
    indx= (raw.field('pmra_err') > 0.)*(raw.field('pmdec_err') > 0.)\
        *(raw.field('vr_err') > 0.)
    raw= raw[indx]
    #Cut on logg?
    if (isinstance(logg,bool) and logg):
        indx= (raw.field('logga') > 3.75)
        raw= raw[indx]
    elif not isinstance(logg,bool):
        indx= (raw.field('logga') > logg)
        raw= raw[indx]
    if isinstance(ug,bool) and ug:
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < 2.)\
            *((raw.field('dered_u')-raw.field('dered_g')) > .6)
        raw= raw[indx]
    if not isinstance(ug,bool):
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < ug[1])\
            *((raw.field('dered_u')-raw.field('dered_g')) > ug[0])
        raw= raw[indx]
    if isinstance(ri,bool) and ri:
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < .4)\
            *((raw.field('dered_r')-raw.field('dered_i')) > -.1)
        raw= raw[indx]
    elif not isinstance(ri,bool):
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < ri[1])\
            *((raw.field('dered_r')-raw.field('dered_i')) > ri[0])
        raw= raw[indx]
    if (isinstance(sn,bool) and sn):
        indx= (raw.field('sna') > 15.)
        raw= raw[indx]
    elif not isinstance(sn,bool):
        indx= (raw.field('sna') > sn)
        raw= raw[indx]
    if isinstance(ebv,bool) and ebv:
        indx= (raw.field('ebv') < .3)
        raw= raw[indx]
    elif not isinstance(ebv,bool):
        indx= (raw.field('ebv') < ebv)
        raw= raw[indx]
    if nocoords: return raw
    raw= _add_distances(raw)
    raw= _add_velocities(raw)
    return raw

def read_kdwarfs(file=_KDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=True,nocoords=False):
    """
    NAME:
       read_kdwarfs
    PURPOSE:
       read the spectroscopic K dwarf sample
    INPUT:
       logg= if True, cut on logg
       ug= if True, cut on u-g
       ri= if True, cut on r-i
       sn= if False, don't cut on SN
       ebv= if True, cut on E(B-V)
       nocoords= if True, don't calculate distances or transform coordinates
    OUTPUT:
       cut data, returns numpy.recarray
    HISTORY:
       2011-07-11 - Written - Bovy@MPIA (NYU)
    """
    raw= _load_fits(file)
    #First cut on r
    indx= (raw.field('dered_r') < 19.)*(raw.field('dered_r') > 14.5)
    raw= raw[indx]
    #Then cut on g-r
    indx= ((raw.field('dered_g')-raw.field('dered_r')) < 0.75)\
        *((raw.field('dered_g')-raw.field('dered_r')) > .55)
    raw= raw[indx]
    #Cut on velocity errs
    indx= (raw.field('pmra_err') > 0.)*(raw.field('pmdec_err') > 0.)\
        *(raw.field('vr_err') > 0.)
    raw= raw[indx]
    #Cut on logg?
    if isinstance(logg,bool) and logg:
        indx= (raw.field('logga') > 3.75)
        raw= raw[indx]
    elif not isinstance(logg,bool):
        indx= (raw.field('logga') > logg)
        raw= raw[indx]
    if isinstance(ug,bool) and ug:
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < 2.5)\
            *((raw.field('dered_u')-raw.field('dered_g')) > 1.5)
        raw= raw[indx]
    elif not isinstance(ug,bool):
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < ug[1])\
            *((raw.field('dered_u')-raw.field('dered_g')) > ug[0])
        raw= raw[indx]
    if isinstance(ri,bool) and ri:
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < .7)\
            *((raw.field('dered_r')-raw.field('dered_i')) > .1)
        raw= raw[indx]
    elif not isinstance(ri,bool):
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < ri[1])\
            *((raw.field('dered_r')-raw.field('dered_i')) > ri[0])
        raw= raw[indx]
    if isinstance(sn,bool) and sn:
        indx= (raw.field('sna') > 15.)
        raw= raw[indx]
    elif not isinstance(sn,bool):
        indx= (raw.field('sna') > sn)
        raw= raw[indx]
    if isinstance(ebv,bool) and ebv:
        indx= (raw.field('ebv') < .3)
        raw= raw[indx]
    elif not isinstance(ebv,bool):
        indx= (raw.field('ebv') < ebv)
        raw= raw[indx]
    if nocoords: return raw
    raw= _add_distances(raw)
    raw= _add_velocities(raw)
    return raw

def _add_distances(raw):
    """Add distances"""
    ds,derrs= ivezic_dist_gr(raw.dered_g,raw.dered_r,raw.feh,
                             return_error=True,dg=raw.g_err,
                             dr=raw.r_err,dfeh=raw.feh_err)
    raw= _append_field_recarray(raw,'dist',ds)
    raw= _append_field_recarray(raw,'dist_err',derrs)
    return raw

def _add_velocities(raw):
    if not _COORDSLOADED:
        print "galpy.util.bovy_coords failed to load ..."
        print "Install galpy for coordinate transformations ..."
        print "*not* adding velocities ..."
        return raw
    #We start from RA and Dec
    lb= bovy_coords.radec_to_lb(raw.ra,raw.dec,degree=True)
    XYZ= bovy_coords.lbd_to_XYZ(lb[:,0],lb[:,1],raw.dist,degree=True)
    pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(raw.pmra,raw.pmdec,
                                                raw.ra,raw.dec,degree=True)
    #print numpy.mean(pmllpmbb[:,0]-raw.pml), numpy.std(pmllpmbb[:,0]-raw.pml)
    #print numpy.mean(pmllpmbb[:,1]-raw.pmb), numpy.std(pmllpmbb[:,1]-raw.pmb)
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(raw.vr,pmllpmbb[:,0],
                                             pmllpmbb[:,1],lb[:,0],lb[:,1],
                                             raw.dist,degree=True)
    #Solar motion from Schoenrich & Binney
    vxvyvz[:,0]+= -11.1
    vxvyvz[:,1]+= 12.24
    vxvyvz[:,2]+= 7.25
    #print numpy.mean(vxvyvz[:,2]), numpy.std(vxvyvz[:,2])
    #Propagate uncertainties
    ndata= len(raw.ra)
    cov_pmradec= numpy.zeros((ndata,2,2))
    cov_pmradec[:,0,0]= raw.pmra_err**2.
    cov_pmradec[:,1,1]= raw.pmdec_err**2.
    cov_pmllbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec,raw.ra,
                                                      raw.dec,degree=True)
    cov_vxvyvz= bovy_coords.cov_dvrpmllbb_to_vxyz(raw.dist,
                                                  raw.dist_err,
                                                  raw.vr_err,
                                                  pmllpmbb[:,0],pmllpmbb[:,1],
                                                  cov_pmllbb,lb[:,0],lb[:,1],
                                                  degree=True)
    #Cast
    XYZ= XYZ.astype(numpy.float64)
    vxvyvz= vxvyvz.astype(numpy.float64)
    cov_vxvyvz= cov_vxvyvz.astype(numpy.float64)
    #Append results to structure
    raw= _append_field_recarray(raw,'xc',XYZ[:,0])
    raw= _append_field_recarray(raw,'yc',XYZ[:,1])
    raw= _append_field_recarray(raw,'zc',XYZ[:,2])
    raw= _append_field_recarray(raw,'vxc',vxvyvz[:,0])
    raw= _append_field_recarray(raw,'vyc',vxvyvz[:,1])
    raw= _append_field_recarray(raw,'vzc',vxvyvz[:,2])
    raw= _append_field_recarray(raw,'vxc_err',numpy.sqrt(cov_vxvyvz[:,0,0]))
    raw= _append_field_recarray(raw,'vyc_err',numpy.sqrt(cov_vxvyvz[:,1,1]))
    raw= _append_field_recarray(raw,'vzc_err',numpy.sqrt(cov_vxvyvz[:,2,2]))
    raw= _append_field_recarray(raw,'vxvyc_rho',cov_vxvyvz[:,0,1]\
                                    /numpy.sqrt(cov_vxvyvz[:,0,0])\
                                    /numpy.sqrt(cov_vxvyvz[:,1,1]))
    raw= _append_field_recarray(raw,'vxvzc_rho',cov_vxvyvz[:,0,2]\
                                    /numpy.sqrt(cov_vxvyvz[:,0,0])\
                                    /numpy.sqrt(cov_vxvyvz[:,2,2]))
    raw= _append_field_recarray(raw,'vyvzc_rho',cov_vxvyvz[:,1,2]\
                                    /numpy.sqrt(cov_vxvyvz[:,1,1])\
                                    /numpy.sqrt(cov_vxvyvz[:,2,2]))
    return raw

def _load_fits(file,ext=1):
    """Loads fits file's data and returns it as a numpy.recarray with lowercase field names"""
    hdulist= pyfits.open(file)
    out= hdulist[ext].data
    hdulist.close()
    return _as_recarray(out)

def _append_field_recarray(recarray, name, new):
    new = numpy.asarray(new)
    newdtype = numpy.dtype(recarray.dtype.descr + [(name, new.dtype)])
    newrecarray = numpy.recarray(recarray.shape, dtype=newdtype)
    for field in recarray.dtype.fields:
        newrecarray[field] = recarray.field(field)
    newrecarray[name] = new
    return newrecarray

def _as_recarray(recarray):
    """go from FITS_rec to recarray"""
    newdtype = numpy.dtype(recarray.dtype.descr)
    newdtype.names= tuple([n.lower() for n in newdtype.names])
    newrecarray = numpy.recarray(recarray.shape, dtype=newdtype)
    for field in recarray.dtype.fields:
        newrecarray[field.lower()] = recarray.field(field)
    return newrecarray

#Ivezic and Juric distance functions
def _mr_gi(gi,feh,dgi=False,dfeh=False):
    """Ivezic+08 photometric distance"""
    if dgi:
        return 14.32-2.*12.97*gi+3.*6.127*gi**2.-4.*1.267*gi**3.\
               +5.*0.0967*gi**4.
    elif dfeh:
        return -1.11-0.36*feh
    else:
        mro= -5.06+14.32*gi-12.97*gi**2.+6.127*gi**3.-1.267*gi**4.\
             +0.0967*gi**5.
        dmr= 4.5-1.11*feh-0.18*feh**2.
        mr= mro+dmr
        return mr

def _mr_ri_bright(ri,dri=False):
    """Juric+08 bright photometric distance"""
    if dri:
        return 13.3-2.*11.5*ri+3.*5.4*ri**2.-4.*0.7*ri**3.
    else:
        return 3.2+13.3*ri-11.5*ri**2.+5.4*ri**3.-0.7*ri**4.

def _mr_ri_faint(ri,dri=False):
    """Juric+08 faint photometric distance"""
    if dri:
        return 11.86-2.*10.74*ri+3.*5.99*ri**2.-4.*1.2*ri**3.
    else:
        return 4.+11.86*ri-10.74*ri**2.+5.99*ri**3.-1.2*ri**4.

def _gi_gr(gr,dr=False,dg=False):
    """(g-i) = (g-r)+(r-i), with Juric et al. (2008) stellar locus for g-r,
    BOVY: JUST USES LINEAR APPROXIMATION VALID FOR < M0"""
    if dg:
        return 1.+1./2.34
    elif dr:
        return -1.-1./2.34
    else:
        ri= (gr-0.12)/2.34
        return gr+ri

def _ri_gr(gr,dr=False,dg=False):
    """(r-i) = f(g-r), with Juric et al. (2008) stellar locus for g-r,
    BOVY: JUST USES LINEAR APPROXIMATION VALID FOR < M0"""
    if dg:
        return 1./2.34
    elif dr:
        return 1./2.34
    else:
        ri= (gr-0.07)/2.34
        return ri


############################CLEAN UP PHOTOMETRY################################
def _cleanup_photometry():
    #Load plates
    platestr= _load_fits(os.path.join(_SEGUESELECTDIR,
                                      'segueplates.fits'))
    plates= list(platestr.plate)
    for ii in range(len(plates)):
        plate= plates[ii]
        platefile= os.path.join(_SEGUESELECTDIR,'segueplates',
                                '%i.fit' % plate)
        try:
            platephot= _load_fits(platefile)
        except AttributeError:
            continue
        #Split into bright and faint
        if 'faint' in platestr[ii].programname:
            indx= (platephot.field('r') >= 17.8)
            platephot= platephot[indx]
        else:
            indx= (platephot.field('r') < 17.8)
            platephot= platephot[indx]
        #Save
        pyfits.writeto(platefile,platephot,clobber=True)
    
#########################ADD KS VALUES TO PLATES###############################
def _add_ks(outfile,sample='g',select='all'):
    """Add the KS probability to the segueplates file"""
    #Load plates
    platestr= _load_fits(os.path.join(_SEGUESELECTDIR,
                                      'segueplates.fits'))
    plates= list(platestr.plate)
    #Load selection functions
    sfconst= segueSelect(sn=True,sample=sample,
                         type_bright='constant',
                         type_faint='constant',select=select)
    sfr= segueSelect(sn=True,sample=sample,
                     type_bright='r',
                     type_faint='r',select=select,
                     dr_bright=0.05,dr_faint=0.2,
                     robust_bright=True)
    if sample.lower() == 'k' and select.lower() == 'program':
        dr_bright= 0.4
        dr_faint= 0.5
    else:
        dr_bright= 0.2
        dr_faint= 0.2
    sfplatesn_r= segueSelect(sn=True,sample=sample,
                             type_bright='platesn_r',
                             type_faint='platesn_r',select=select,
                             dr_bright=dr_bright,
                             dr_faint=dr_faint,
                             robust_bright=True)
    sfsharp= segueSelect(sn=True,sample=sample,
                         type_bright='sharprcut',
                         type_faint='sharprcut',select=select)
    sftanh= segueSelect(sn=True,sample=sample,
                        type_bright='tanhrcut',
                        type_faint='tanhrcut',select=select)
    #Calculate KS for each plate
    nplates= len(plates)
    ksconst= numpy.zeros(nplates)
    ksr= numpy.zeros(nplates)
    ksplatesn_r= numpy.zeros(nplates)
    kssharp= numpy.zeros(nplates)
    kstanh= numpy.zeros(nplates)
    for ii in range(nplates):
        plate= plates[ii]
        sys.stdout.write('\r'+"Working on plate %i" % plate)
        sys.stdout.flush()
        try:
            ksconst[ii]= sfconst.check_consistency(plate)
        except KeyError:
            continue
        ksr[ii]= sfr.check_consistency(plate)
        ksplatesn_r[ii]= sfplatesn_r.check_consistency(plate)
        kssharp[ii]= sfsharp.check_consistency(plate)
        kstanh[ii]= sftanh.check_consistency(plate)
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    #Add to platestr
    platestr= _append_field_recarray(platestr,'ksconst_'+sample+'_'+select,
                                     ksconst)
    platestr= _append_field_recarray(platestr,'ksr_'+sample+'_'+select,
                                     ksr)
    platestr= _append_field_recarray(platestr,'ksplatesn_r_'+sample+'_'+select,
                                     ksplatesn_r)
    platestr= _append_field_recarray(platestr,'kssharp_'+sample+'_'+select,
                                     kssharp)
    platestr= _append_field_recarray(platestr,'kstanh_'+sample+'_'+select,
                                     kstanh)
    #Save
    pyfits.writeto(outfile,platestr,clobber=True)
    return
