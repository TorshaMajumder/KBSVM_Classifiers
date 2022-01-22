
#
# Import all the dependencies
#


import numpy as np


def redshift_spectrum(wave, z):

    """
        This method is used for redshifting the spectrum to its rest frame

        ...

        Parameters
        ----------

        wave: float
            wavelength of the spectrum
        z: float
            redshifting


        Returns
        -------
        wave: float
            redshifted wavelength of the spectrum
          

        """
    
    wave_new = wave * (z + 1)
    return wave_new


def deredshift_spectrum(wave, z):

    """
    This method is used for de-redshifting the spectrum to its rest frame

    ...

    Parameters
    ----------

    wave: float
        wavelength of the spectrum
    z: float
     redshifting


    Returns
    -------
    wave: float
        de-redshifted wavelength of the spectrum
          

    """

    wave_new = np.array(wave) / (z + 1)

    return (wave_new)


def min_max_index(flux, outerVal=0):

    """

        This method is used to find the index of the minimum and maximum binned flux

        ...

        Parameters
        ----------
        flux: float
            flux 
        outerVal: float (default = 0.0)
            the points in the spectrum that do not 
            have data in the range w0 to w1 is set to outerVal
            

        Returns
        -------
        minIndex: int
            index of the minimum binned_flux
        maxIndex: int
            index of the maximum binned_flux 
            


    """

    nonZeros = np.where(flux != outerVal)[0]

    if nonZeros.size:
        minIndex, maxIndex = min(nonZeros), max(nonZeros)
    else:
        minIndex, maxIndex = len(flux), len(flux)

    return (minIndex, maxIndex)

    
def mean_zero(flux, minIndex, maxIndex):

    """

        This method is used to return a zero mean flux

        
        ...

        Parameters
        ----------
        flux: float
            flux 
        minIndex: int
            index of the minimum binned_flux
        maxIndex: int
            index of the maximum binned_flux 
        
            

        Returns
        -------
        meanzeroflux: float
            mean zero flux
            
    """
    
    meanflux = np.mean(flux[minIndex:maxIndex])
    meanzeroflux = flux - meanflux
    meanzeroflux[0:minIndex] = flux[0:minIndex]
    meanzeroflux[maxIndex + 1:] = flux[maxIndex + 1:]

    return meanzeroflux



def limit_wavelength_range(wave, flux, minWave, maxWave):

    """

        This method is used to set the flux value to zero beyond the minWave-maxWave range

        
        ...

        Parameters
        ----------
        wave: float
            input wave 
        flux: float
            input flux 
        minWave: float
            minimum wavelength
        maxWave: float
            maximum wavelength
        
            

        Returns
        -------
        flux: float
          flux
            
    """
    minIdx = (np.abs(wave - minWave)).argmin()
    maxIdx = (np.abs(wave - maxWave)).argmin()
    flux[:minIdx] = np.zeros(minIdx)
    flux[maxIdx:] = np.zeros(len(flux) - maxIdx)

    return flux

def two_col_input_spectrum(wave, flux, z, w0, w1):
        
        wave = deredshift_spectrum(wave , z)
        
        mask = (wave >= w0) & (wave < w1)
        
        wave = wave[mask]
        flux = flux[mask]

        if not wave.any():
            raise Exception("The spectrum file with redshift {1} is out of the wavelength range {2}A to {3}A, "
                            "and cannot be classified. Please remove this object or change the input redshift of this"
                            " spectrum.".format(z, int(w0), int(w1)))

        fluxNorm = (flux - min(flux)) / (max(flux) - min(flux))

        return wave, fluxNorm


def vectorised_log_binning(wave, flux, w0, w1, nw, dwlog):

    spec = np.array([wave, flux]).T
    mask = (wave >= w0) & (wave < w1)
    spec = spec[mask]
    wave, flux = spec.T
    try:
        fluxOut = np.zeros(int(nw))
        waveMiddle = wave[1:-1]
        waveTake1Index = wave[:-2]
        wavePlus1Index = wave[2:]
        s0List = 0.5 * (waveTake1Index + waveMiddle)
        s1List = 0.5 * (waveMiddle + wavePlus1Index)
        s0First = 0.5 * (3 * wave[0] - wave[1])
        s0Last = 0.5 * (wave[-2] + wave[-1])
        s1First = 0.5 * (wave[0] + wave[1])
        s1Last = 0.5 * (3 * wave[-1] - wave[-2])
        s0List = np.concatenate([[s0First], s0List, [s0Last]])
        s1List = np.concatenate([[s1First], s1List, [s1Last]])
        s0LogList = np.log(s0List / w0) / dwlog + 1
        s1LogList = np.log(s1List / w0) / dwlog + 1
        dnuList = s1List - s0List
        s0LogListInt = s0LogList.astype(int)
        s1LogListInt = s1LogList.astype(int)
        numOfJLoops = s1LogListInt - s0LogListInt
        jIndexes = np.flatnonzero(numOfJLoops)
        jIndexVals = s0LogListInt[jIndexes]
        prependZero = jIndexVals[0] if jIndexVals[0] < 0 else False
        if prependZero is not False:
            jIndexVals[0] = 0
            numOfJLoops[0] += prependZero
        numOfJLoops = (numOfJLoops[jIndexes])[jIndexVals < nw]
        fluxValList = ((flux * 1 / (s1LogList - s0LogList) * dnuList)[jIndexes])[jIndexVals < nw]
        fluxValList = np.repeat(fluxValList, numOfJLoops)
        minJ = min(jIndexVals)
        maxJ = (max(jIndexVals) + numOfJLoops[-1]) if (max(jIndexVals) + numOfJLoops[-1] < nw) else nw
        fluxOut[minJ:maxJ] = fluxValList[:(maxJ - minJ)]

        return fluxOut
    except Exception as e:
        print(e)
        print('\nwave:\n', wave)
        print('\nflux\n', flux)
        print("########################################ERROR#######################################\n\n\n\n")
        return np.zeros(nw) 



def original_log_binning(self, wave, flux):
    """ Rebin wavelengths: adapted from SNID apodize.f subroutine rebin() """
    fluxOut = np.zeros(int(self.nw))

    for i in range(0, len(wave)):
        if i == 0:
            s0 = 0.5 * (3 * wave[i] - wave[i + 1])
            s1 = 0.5 * (wave[i] + wave[i + 1])
        elif i == len(wave) - 1:
            s0 = 0.5 * (wave[i - 1] + wave[i])
            s1 = 0.5 * (3 * wave[i] - wave[i - 1])
        else:
            s0 = 0.5 * (wave[i - 1] + wave[i])
            s1 = 0.5 * (wave[i] + wave[i + 1])

        s0log = np.log(s0 / self.w0) / self.dwlog + 1
        s1log = np.log(s1 / self.w0) / self.dwlog + 1
        dnu = s1 - s0

        for j in range(int(s0log), int(s1log)):
            if j < 0 or j >= self.nw:
                continue
            alen = 1  # min(s1log, j+1) - max(s0log, j)
            fluxval = flux[i] * alen / (s1log - s0log) * dnu
            fluxOut[j] = fluxOut[j] + fluxval

    return fluxOut

