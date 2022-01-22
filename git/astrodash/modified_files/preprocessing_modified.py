#
# Import all the dependencies
#

import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from read_file import ReadSpectrumFile
from array_tools import normalise_spectrum, zero_non_overlap_part
from preprocessing_tools import min_max_index, vectorised_log_binning, mean_zero, limit_wavelength_range





class PreProcessSpectrum(object):

   """

    Preprocessing algorithm utilized by DASH
    
    ...

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

    
    Methods
    -------

        1. median_filter():
            return: filtered flux after passing through median filter (preFiltered)

        2. deredshifting():
            returns: de-redshifted wavelength of the spectrum (wave),
                     de-redshifted flux of the spectrum (deredshifted)

        3. log_wavelength():
            return: log-wavelength axis (wlog), 
                    binned_flux (fluxOut), 
                    index of the minimum binned_flux (minIndex), 
                    index of the maximum binned_flux (maxIndex)

        4. spline_fit():
            returns: continuum (continuum)

        5. continuum_removal():
            returns: normalized flux (contRemovedFluxNorm),
                     continuum (new_continuum)
                    
        6. apodize():
            returns: output flux (fluxout)
            

    
    
    
    Note
    ----
    1. The attributes can be changed by a user who wishes to re-train the CNN model. 
       However, the default parameters are nw = 1024, w0 = 3500, w1 = 10000, which
       covers the optical spectral range at which most supernova events are observed.

    2. Input spectra in DASH have a default smoothing factor of smooth = 6, but can 
       be altered by the user.
    
    3. A 13-point cubic spline interpolation is used to model the continuum. 
       13 points was considered to be suficient to interpolate the spectrum.
    
       
   """



   def __init__(self, filename=None, w0=3500, w1=10000, nw=1024, z=0.5):

        """

        Constructs all the necessary attributes for Preprocessing the data
        ...

        Parameters
        ----------
        filename: file
            input file
        z: float
            redshifting
        w0 : float
            minimum binned wavelength
        w1: float
            maximum binned wavelength
        nw: float
            number of fixed points in the log-wavelength binning
       
        """
        self.filename = filename
        self.z = z
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        
        #
        # dwlog: size of a logarithmic wavelength bin
        #
        self.dwlog = np.log(w1 / w0) / nw
        


   
   def median_filter(self, wave, flux, smooth=6):
    
       """
        This method is used to apply a low-pass median filter to each spectrum in order to remove high-frequency noise and cosmic rays
        
        ...

        Parameters
        ----------
        wave : float
            input wavelength
        flux: float
            input flux
        smooth: float (default = 6)
            smoothing factor


        Returns
        -------
        preFiltered: float
            filtered flux after passing through low-pass median filter 
            

       """
       #
       # average wavelength spacing of the spectrum
       # min(self.wave) and max(self.wave) are the minimum and maximum wavelength of the spectrum, and 
       # len(self.wave) is the number of points in the spectrum
       #
       wavelengthDensity = (max(wave) - min(wave)) / len(wave)
       # 
       # wavelength density of the final spectra after processing 
       #
       self.wDensity = (self.w1 - self.w0) / self.nw  
       #
       # window size of the median filter
       #
       filterSize = int(self.wDensity / wavelengthDensity * smooth / 2) * 2 + 1
       #
       # filtered flux after passing through median filter 
       #
       filtered_flux = medfilt(flux, kernel_size=filterSize)
        
       return(wave, filtered_flux)


   def deredshifting(self, wave):

        """
        This method is used for de-redshifting the spectrum to its rest frame

        ...

        Parameters
        ----------

        wave: float
            wavelength of the spectrum
        

        Returns
        -------
        wave: float
            de-redshifted wavelength of the spectrum
          

        """

        wave_new = np.array(wave) / (self.z + 1)

        return (wave_new)




   def log_wavelength(self, wave, flux):

        """
        It returns the log-wavelength axis, binned_flux, index of the minimum binned_flux, and index of the maximum binned_flux

        ...

        Parameters
        ----------
        wave: float
            wavelength of the spectrum
        flux: float
            flux of the spectrum


        Returns
        ------
        wlog: log-wavelength axis
        fluxOut: binned_flux 
        minIndex: index of the minimum binned_flux
        maxIndex: index of the maximum binned_flux 

        """
        #
        # Calculate the log-wavelength axis from index n = 0 to nw
        #
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * self.dwlog)
        wlog = np.array(wlog)
        #
        # Calculate the binned-flux using the vectorized_log_binning method
        #
        fluxOut = vectorised_log_binning(wave, flux, self.w0, self.w1, self.nw, self.dwlog)
        #
        # Calculate index of the - minimum binned_flux and maximum binned_flux 
        #
        minIndex, maxIndex = min_max_index(fluxOut, outerVal=0)

        return (wlog, fluxOut, minIndex, maxIndex) 



   
   def spline_fit(self, wave, flux, minIndex, maxIndex, numSplinePoints=13):
        
        """

        This method is used in preparing a 13-point cubic spline interpolation 
        which is used to model the continuum.

        ...

        Parameters
        ----------
        wave: float
            log-wavelength axis
        flux: float
            binned_flux 
        minIndex: int
            index of the minimum binned_flux
        maxIndex: int
            index of the maximum binned_flux 
        numSplinePoints: int (default = 13)
            13-point cubic spline interpolation
            

        Returns
        -------
        continuum: float
            continuum


        """
        #
        # Initilize the continuum
        #
        continuum = np.zeros(int(self.nw)) + 1
        #
        # Consider at least 6 points in the spectrum
        #
        if (maxIndex - minIndex) > 5:
            #
            # Fit a cubic spline
            #
            spline = UnivariateSpline(wave[minIndex:maxIndex + 1], flux[minIndex:maxIndex + 1], k=3)
            #
            # Divide the wavelengths in 13 points
            #
            splineWave = np.linspace(wave[minIndex], wave[maxIndex], num=numSplinePoints, endpoint=True)
            #
            # 13-point cubic spline interpolation
            #
            splinePoints = spline(splineWave)
            #
            # A 13-point cubic spline interpolation is used to model the continuum
            #
            splineMore = UnivariateSpline(splineWave, splinePoints, k=3)
            splinePointsMore = splineMore(wave[minIndex:maxIndex + 1])
            continuum[minIndex:maxIndex + 1] = splinePointsMore
        else:
            print("WARNING: Less than 6 points in the spectrum")

        return continuum 



   def continuum_removal(self, wave, flux, minIndex, maxIndex, numSplinePoints=13):

        """

        This method is used in preparing the spectra that involves dividing the continuum. 
        A 13-point cubic spline interpolation is used to model the continuum.

        ...

        Parameters
        ----------
        wave: float
            log-wavelength axis
        flux: float
            binned_flux 
        minIndex: int
            index of the minimum binned_flux
        maxIndex: int
            index of the maximum binned_flux 
        numSplinePoints: int (default = 13)
            13-point cubic spline interpolation
            

        Returns
        -------
        contRemovedFluxNorm: float
        new_continuum: float
            


        """
        
        #
        # Remove any zeroed flux
        #
        flux = flux + 1  
        #
        #
        #
        contRemovedFlux = np.copy(flux)
        #
        # A 13-point cubic spline interpolation is used to model the continuum
        #
        continuum = self.spline_fit(wave, flux, minIndex, maxIndex, numSplinePoints)
        #
        # Continuum division: the continuum is then divided from the spectrum
        #
        contRemovedFlux[minIndex:maxIndex + 1] = flux[minIndex:maxIndex + 1] / continuum[minIndex:maxIndex + 1]
        #
        # Normalize the spectra 
        #
        contRemovedFluxNorm = normalise_spectrum(contRemovedFlux - 1)
        #
        # Set the values to outerVal outside the minIndex-maxindex index range
        #
        contRemovedFluxNorm = zero_non_overlap_part(contRemovedFluxNorm, minIndex, maxIndex)
        #
        # Adjust the continuum
        #
        new_continuum = continuum - 1

        return (contRemovedFluxNorm, new_continuum)
    


   def apodize(self, flux, minIndex, maxIndex, outerVal=0.0):

        """

        This method is used to remove the discontinuities in each end of the spectra by apodizing the
        spectrum with a cosine bell. This involves multiplying 5% of each end of the spectrum by a cosine, 
        to remove sharp spikes.

        
        ...

        Parameters
        ----------
        flux: float
            mean zero flux 
        minIndex: int
            index of the minimum binned_flux
        maxIndex: int
            index of the maximum binned_flux 
        outerVal: float (default = 0.0)
            the points in the spectrum that do not 
            have data in the range w0 to w1 is set to outerVal
            

        Returns
        -------
        fluxout: float
            output flux
            


        """


        #
        # apodize with 5% cosine bell
        #
        percent = 0.05
        fluxout = np.copy(flux) - outerVal
        #
        # multiply 5% of each end of the spectrum by a cosine, to remove sharp spikes.
        #
        nsquash = int(self.nw * percent)
        for i in range(0, nsquash):
            arg = np.pi * i / (nsquash - 1)
            factor = 0.5 * (1 - np.cos(arg))
            if (minIndex + i < self.nw) and (maxIndex - i >= 0):
                fluxout[minIndex + i] = factor * fluxout[minIndex + i]
                fluxout[maxIndex - i] = factor * fluxout[maxIndex - i]
            else:
                print("Invalid flux in APODIZE()")
                print("minIndex=%d, i=%d" % (minIndex, i))
                break
        #
        # Adjust the flux
        #
        if outerVal != 0:
            fluxout = fluxout + outerVal
            #
            # Set the values to outerVal outside the minIndex-maxindex index range
            #
            fluxout = zero_non_overlap_part(fluxout, minIndex, maxIndex, outerVal=outerVal)

        return fluxout


   def preprocess_spectrum(self, filename, w0, w1, nw):

    #
    # Read the input file
    #
    read_file=ReadSpectrumFile(filename, w0, w1, nw, self.z)
    wave, flux = read_file.read_file_type()
    wave, flux = np.array(wave), np.array(flux)
    #
    # Normalize the spectrum
    #
    flux = normalise_spectrum(flux)
    #
    # Find the flux based on the minimum and maximum wavelngth
    #
    flux = limit_wavelength_range(wave, flux, w0, w1)
    #
    # Preprocess the spectrum
    #
    preprocess=PreProcessSpectrum(filename, self.z, w0, w1, nw)
    wave, flux = preprocess.median_filter(wave, flux)
    wave = preprocess.deredshifting(wave)
    wave, flux, minIndex, maxIndex = preprocess.log_wavelength(wave, flux)
    flux, continuum = preprocess.continuum_removal(wave, flux, minIndex, maxIndex)
    new_flux = mean_zero(flux, minIndex, maxIndex)
    apodized = preprocess.apodize(new_flux, minIndex, maxIndex)
    norm_flux = normalise_spectrum(apodized)
    new_spectrum = zero_non_overlap_part(norm_flux, minIndex, maxIndex, outerVal=0.5)

    
    return(wave, new_spectrum, minIndex, maxIndex)


        
     






        