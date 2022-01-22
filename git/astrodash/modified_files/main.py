import numpy as np
from astrodash.array_tools import zero_non_overlap_part
from read_file import ReadSpectrumFile
from array_tools import normalise_spectrum
from preprocessing_modified import PreProcessSpectrum
from preprocessing_tools import mean_zero, limit_wavelength_range



def preprocess_spectrum(filename, w0, w1, nw, z):

    #
    # Read the input file
    #
    read_file=ReadSpectrumFile(filename, w0, w1, nw, z)
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
    preprocess=PreProcessSpectrum(filename, z, w0, w1, nw)
    wave, flux = preprocess.median_filter(wave, flux)
    wave = preprocess.deredshifting(wave)
    wave, flux, minIndex, maxIndex = preprocess.log_wavelength(wave, flux)
    flux, continuum = preprocess.continuum_removal(wave, flux, minIndex, maxIndex)
    new_flux = mean_zero(flux, minIndex, maxIndex)
    apodized = preprocess.apodize(new_flux, minIndex, maxIndex)
    norm_flux = normalise_spectrum(apodized)
    new_spectrum = zero_non_overlap_part(norm_flux, minIndex, maxIndex, outerVal=0.5)

    
    return(new_spectrum)





if __name__ == '__main__':

    
    filename= './dataset/test/DES16X3bdj.fits'
    w0 = 3500
    w1 = 10000
    nw = 1024
    z = 0.24

    spectrum = preprocess_spectrum(filename, w0, w1, nw, z)
    print(spectrum)

    


    
