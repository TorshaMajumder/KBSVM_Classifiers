
#
# Import all the dependencies
#
import re
import os
import numpy as np
import pandas as pd
from astropy.io import fits



class ReadSpectrumFile(object):

    """
    Attributes
    ----------
    filename: file
        input file
    w0 : float
        minimum binned wavelength
    w1: float
        maximum binned wavelength
    nw: float
        number of fixed points in the log-wavelength binning
    z: float
        redshifting
    """
    def __init__(self, filename=None, w0=3500, w1=10000, nw=1024, z=0.5):
        #
        # input file
        #
        self.filename = filename
        #
        # valid file format
        #
        self.two_col_ftype=['.dat', '.txt', '.flm',  '.csv']
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.z = z
        
    
    
    def read_file_type(self,template=False):

        #
        # use regex to parse the extension of the file format
        #

        try:
            #print(self.filename)
            if self.filename:
                #print("11111::::", self.filename)
                #template=False
                fname = os.path.basename(self.filename)
                extension = self.filename.split('.')[-1]

                if template and extension == 'dat' and len(fname.split('.')) == 3 and fname.split('.')[1][0] in ['m', 'p']:
                    # fname = os.path.basename(self.filename)
                    # extension = self.filename.split('.')[-1]

                    #if extension == 'dat' and len(fname.split('.')) == 3 and fname.split('.')[1][0] in ['m', 'p']:  # Check if input is a superfit template
                    
                    wave, flux = self.read_two_col_ftype()
                    tType = os.path.split(os.path.split(self.filename)[0])[-1]  # Name of directory is the type name
                    #fname = os.path.basename(self.filename)
                    snName, ageInfo = os.path.basename(fname).strip('.dat').split('.')
                    
                    if ageInfo == 'max': age = 0
                    elif ageInfo[0] == 'm': age = -float(ageInfo[1:])
                    elif ageInfo[0] == 'p': age = float(ageInfo[1:])
                    else:
                        raise Exception("Invalid Superfit file: {0}".format(self.filename))

                    nCols = 1

                    return wave, flux, nCols, [age], tType



                else:
                    f_type = re.search('\.[dtlfDTLF]*?\w+', self.filename)
                    if f_type:
                        f_type=f_type.group(0)
                        f_type = f_type.lower()
                        #
                        # check for files in the two_col_ftype list
                        #
                        if (f_type in self.two_col_ftype):
                            #print(f"\n{f_type} is a known file type")
                            wave, flux = self.read_two_col_ftype()
                            return wave, flux

                        #
                        # check for .lnw files
                        #
                        elif (f_type == '.lnw'):
                            #print(f"\n{f_type} is a known file type")
                            wave, flux, numAges, ages, ttype, splineInfo = self.read_lnw_ftype()
                            return wave, flux, numAges, ages, ttype, splineInfo
                
                        #
                        # check for .fits files
                        #
                        elif(f_type == '.fits'):
                            #print(f"\n{f_type} is a known file type")
                            wave, flux = self.read_fits_ftype()
                            return wave, flux
                        #
                        # check for incompatible files
                        #
                        else:
                            print(f"\n{f_type} is not compatible with ASTRODASH.\n\nPlease insert a new file!")

                    #
                    # check for files without any file extension
                    #
                    else:
                        #print(f"\n{f_type} is an unknown file type.\n\nLet's validate the file......")
                        wave, flux = self.read_two_col_ftype()
                        #print(wave,flux)
                        return wave, flux

        except AttributeError:
            print(f"\nCouldn't find any file type for: {self.filename}")

        

    
    
    def read_two_col_ftype(self):
        
        try:
            data = pd.read_csv(self.filename, header=None, sep=';|,|\s+', engine='python')
            
            data.dropna(axis=1, inplace=True)
            data=data.T.reset_index(drop=True).T
            if (len(data.columns) >1 and len(data.columns) <=3):
            
                wave = data.iloc[:,0]
                flux = data.loc[:,1]
                wave, flux = np.array(wave), np.array(flux)
                #print(f"\nThe file is parsed successfully!\n\nWavelength and Flux are extracted.")
                return wave, flux

        except Exception:
            print(f"\nPlease check the file {self.filename}!\nThe file should contain two-three columns.")


    def read_fits_ftype(self):

        hdu = fits.open(self.filename)
        flux = hdu[0].data
        header = hdu[0].header
        if 'CDELT1' in header:
            wave_step = header['CDELT1']
        else:
            wave_step = header['CD1_1']
        wave_start = header['CRVAL1'] - (header['CRPIX1'] - 1) * wave_step
        wave_num = flux.shape[0]
        wave = list(np.linspace(wave_start, wave_start + wave_step * wave_num, num=wave_num))
        flux[np.isnan(flux)] = 0  
        hdu.close()
        wave, flux = np.array(wave), np.array(flux)
        return wave, flux
    

    
    def read_lnw_ftype(self):

        lnw_info = dict()

        with open(self.filename, 'r') as FileObj:
            for lineNum, line in enumerate(FileObj):
                if lineNum == 0:
                    header = (line.strip('\n')).split(' ')
                    header = [x for x in header if x != '']
                    numAges, nwx, w0x, w1x, mostKnots, tname, dta, ttype, ittype, itstype = header
                    numAges, mostKnots = map(int, (numAges, mostKnots))
                    nk = np.zeros(numAges)
                    fmean = np.zeros(numAges)
                    xk = np.zeros((mostKnots, numAges))
                    yk = np.zeros((mostKnots, numAges))
                    
                elif lineNum == 1:
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        nk[j], fmean[j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])
                elif lineNum in range(2, mostKnots + 2):
                    splineInfo = (line.strip('\n')).split(' ')
                    splineInfo = [x for x in splineInfo if x != '']
                    for j in range(numAges):
                        xk[lineNum - 2, j], yk[lineNum - 2, j] = (splineInfo[2 * j + 1], splineInfo[2 * j + 2])

                elif lineNum == mostKnots + 2:
                    break

        lnw_info['numAges']= numAges
        lnw_info['nwx']= nwx
        lnw_info['w0x']= w1x
        lnw_info['mostKnots']= mostKnots
        lnw_info['tname']= tname
        lnw_info['dta']= dta
        lnw_info['ttype']= ttype
        lnw_info['ittype']= ittype
        lnw_info['itstype']= itstype
        lnw_info['nk']= nk
        lnw_info['fmean']= fmean
        lnw_info['xk']= pd.DataFrame(xk)
        lnw_info['yk']= pd.DataFrame(yk)

        splineInfo = (nk, fmean, xk, yk)
        

        try:
            data = pd.read_csv(self.filename, skiprows=mostKnots + 2, header=None, sep=';|,|\s+', engine='python')
            data.dropna(axis=1, inplace=True)
            data=data.T.reset_index(drop=True).T
            #if (len(data.columns) >1 and len(data.columns) <=3):
            ages = data.iloc[0,1:]
            data = data.drop(0)
            wave = data.iloc[:,0]
            flux = data.drop(0, axis=1)
            flux = flux.T
            wave, flux, ages = np.array(wave), np.array(flux), np.array(ages)

            if ttype == 'Ia-99aa':
                ttype = 'Ia-91T'
            elif ttype == 'Ia-02cx':
                ttype = 'Iax'
            elif ttype == 'Ic-SL':
                ttype = 'SLSNe'

            #print("11::::", ttype)
            #print(f"\nThe file is parsed successfully!\n\nWavelength and Flux are extracted.")
            return wave, flux, numAges, ages, ttype, splineInfo
        
        except Exception:
            print(f"\nPlease check the file {self.filename}!\nThe file should contain two-three columns.")




            

        


    
    
    
    
    
    
    
    
    
    
