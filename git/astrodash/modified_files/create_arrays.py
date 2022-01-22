import numpy as np
from random import shuffle
import multiprocessing as mp
import itertools
from imblearn import over_sampling
from preprocessing_tools import min_max_index
from helpers import temp_list, div0
from preprocessing_modified import PreProcessSpectrum
#from astrodash.sn_processing import PreProcessing
from combine_sn_and_host import training_template_data
#from astrodash.preprocessing import ProcessingTools
from array_tools import zero_non_overlap_part, normalise_spectrum




class AgeBinning(object):
    def __init__(self, minAge, maxAge, ageBinSize):
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize

    def age_bin(self, age):

        """
        Cutting the age_bin into three fragments: [-20 to -5), [-5 to +10), [+10 to +80] //
                                                [minAge to -5), [-5 to +10), [+10 to maxAge]
        :param age: age for the corresponding bin
        :return: age_bin
        """
        # ageBin = int(round((age-self.minAge) / self.ageBinSize))

        if age >= self.minAge and age < -5.0: ageBin=0
        elif age >= -5.0 and age < 15.0: ageBin=1
        else: ageBin=2

        return ageBin

    def age_labels(self):


        ageLabels = list()
        ageLabels.append(f"{self.minAge} to -5.0")
        ageLabels.append("-5.0 to 15.0")
        ageLabels.append(f" 10.0 to {self.maxAge}")


        # if self.minAge < -18:
        #     ageLabels.append(f"{self.minAge} to -18")
        #     min_age = -18
        #     start = int(min_age + self.ageBinSize)
        #     stop = int(self.maxAge + self.ageBinSize)
        #
        # for i in range(start, stop, int(self.ageBinSize)):
        #     ageLabels.append(f"{min_age} to {i}")
        #     min_age = i

        return ageLabels


class CreateLabels(object):

    def __init__(self, nTypes, minAge, maxAge, ageBinSize, typeList, hostList, nHostTypes):
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        # self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge - 0.1) + 1
        self.numOfAgeBins = 3
        self.nLabels = self.nTypes * self.numOfAgeBins
        self.ageLabels = self.ageBinning.age_labels()
        # print(self.nLabels, self.ageLabels)
        self.hostList = hostList
        self.nHostTypes = nHostTypes


    def label_array(self, ttype, age, host=None):
        ageBin = self.ageBinning.age_bin(age)
        #print(ttype)
        try:
            typeIndex = self.typeList.index(ttype)
        except ValueError as err:
            raise Exception("INVALID TYPE: {0}".format(err))

        if host is None:
            labelArray = np.zeros((self.nTypes, self.numOfAgeBins))
            labelArray[typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = ttype + ": " + self.ageLabels[ageBin]
        else:
            hostIndex = self.hostList.index(host)
            labelArray = np.zeros((self.nHostTypes, self.nTypes, self.numOfAgeBins))
            labelArray[hostIndex][typeIndex][ageBin] = 1
            labelArray = labelArray.flatten()
            typeName = "{}: {}: {}".format(host, ttype, self.ageLabels[ageBin])

        labelIndex = np.argmax(labelArray)

        return labelIndex, typeName

    def type_names_list(self):
        typeNamesList = []
        if self.hostList is None:
            for tType in self.typeList:
                for ageLabel in self.ageBinning.age_labels():
                    typeNamesList.append("{}: {}".format(tType, ageLabel))
        else:
            for host in self.hostList:
                for tType in self.typeList:
                    for ageLabel in self.ageBinning.age_labels():
                        typeNamesList.append("{}: {}: {}".format(host, tType, ageLabel))

        return np.array(typeNamesList)


class ReadSpectra(object):

    def __init__(self, w0, w1, nw, snFilename, galFilename=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.snFilename = snFilename
        self.galFilename = galFilename
        if galFilename is None:
            self.data = PreProcessSpectrum(filename=snFilename, w0=w0, w1=w1, nw=nw)

    def sn_plus_gal_template(self, snAgeIdx, snCoeff, galCoeff, z):
        wave, flux, minIndex, maxIndex, nCols, ages, tType = training_template_data(snAgeIdx, snCoeff, galCoeff, z,
                                                                                    self.snFilename, self.galFilename,
                                                                                    self.w0, self.w1, self.nw)

        return wave, flux, nCols, ages, tType, minIndex, maxIndex

    def input_spectrum(self, z, smooth, minWave, maxWave):
        self.data.z=z
        wave, flux, minIndex, maxIndex= self.data.preprocess_spectrum()

        return wave, flux, int(minIndex), int(maxIndex), self.data.z


class ArrayTools(object):

    def __init__(self, nLabels, nw):
        self.nLabels = nLabels
        self.nw = nw

    def shuffle_arrays(self, memmapName='', **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        arraySize = len(kwargs['labels'])
        if arraySize == 0:
            return kwargs

        kwargShuf = {}
        self.randnum = np.random.randint(10000)
        for key in kwargs:
            if key == 'images':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.float16,
                                      mode='w+', shape=(arraySize, int(self.nw)))
            elif key == 'labels':
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=np.uint16,
                                      mode='w+', shape=arraySize)
            else:
                arrayShuf = np.memmap('shuffled_{}_{}_{}.dat'.format(key, memmapName, self.randnum), dtype=object,
                                      mode='w+', shape=arraySize)
            kwargShuf[key] = arrayShuf

        print("Shuffling...")
        # Randomise order
        p = np.random.permutation(len(kwargs['labels']))
        for key in kwargs:
            assert len(kwargs[key]) == arraySize
            print(key, "shuffling...")
            print(len(p))
            kwargShuf[key] = kwargs[key][p]

        return kwargShuf

    def count_labels(self, labels):
        counts = np.zeros(self.nLabels)

        for i in range(len(labels)):
            counts[labels[i]] += 1

        return counts

    def augment_data(self, flux, stdDevMean=0.05, stdDevStdDev=0.05):
        minIndex, maxIndex = min_max_index(flux, outerVal=0.5)
        noise = np.zeros(self.nw)
        stdDev = abs(np.random.normal(stdDevMean, stdDevStdDev))  # randomised standard deviation
        noise[minIndex:maxIndex] = np.random.normal(0, stdDev, maxIndex - minIndex)
        # # Add white noise to regions outside minIndex to maxIndex
        # noise[0:minIndex] = np.random.uniform(0.0, 1.0, minIndex)
        # noise[maxIndex:] = np.random.uniform(0.0, 1.0, self.nw-maxIndex)

        augmentedFlux = flux + noise
        augmentedFlux = normalise_spectrum(augmentedFlux)
        augmentedFlux = zero_non_overlap_part(augmentedFlux, minIndex, maxIndex, outerVal=0.5)

        return augmentedFlux


class OverSampling(ArrayTools):
    def __init__(self, nLabels, nw, **kwargs):
        """ Must take images and labels as arguments with the keyword specified.
        Can optionally take filenames and typeNames as arguments """
        ArrayTools.__init__(self, nLabels, nw)
        self.kwargs = kwargs

        counts = self.count_labels(self.kwargs['labels'])
        print("Before OverSample")  #
        print(counts)  #

        self.overSampleAmount = np.rint(div0(1 * max(counts), counts))  # ignore zeros in counts
        self.overSampleArraySize = int(sum(np.array(self.overSampleAmount, int) * counts))
        print(np.array(self.overSampleAmount, int) * counts)
        print(np.array(self.overSampleAmount, int))
        print(self.overSampleArraySize, len(self.kwargs['labels']))
        self.kwargOverSampled = {}
        self.randnum = np.random.randint(10000)
        for key in self.kwargs:
            if key == 'images':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.float16,
                                             mode='w+',
                                             shape=(self.overSampleArraySize, int(self.nw)))
            elif key == 'labels':
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=np.uint16,
                                             mode='w+',
                                             shape=self.overSampleArraySize)
            else:
                arrayOverSampled = np.memmap('oversampled_{}_{}.dat'.format(key, self.randnum), dtype=object, mode='w+',
                                             shape=self.overSampleArraySize)
            self.kwargOverSampled[key] = arrayOverSampled

        self.kwargShuf = self.shuffle_arrays(memmapName='pre-oversample_{}'.format(self.randnum), **self.kwargs)
        print(len(self.kwargShuf['labels']))

    def oversample_mp(self, i_in, offset_in, std_in, labelIndex_in):
        print('oversampling', i_in, len(self.kwargShuf['labels']))
        oversampled = {key: [] for key in self.kwargs}
        repeatAmount = int(self.overSampleAmount[labelIndex_in])
        for r in range(repeatAmount):
            for key in self.kwargs:
                if key == 'images':
                    oversampled[key].append(
                        self.augment_data(self.kwargShuf[key][i_in], stdDevMean=0.05, stdDevStdDev=std_in))
                else:
                    oversampled[key].append(self.kwargShuf[key][i_in])
        return oversampled, offset_in, repeatAmount

    def collect_results(self, result):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        oversampled_in, offset_in, repeatAmount = result
        for key in self.kwargs:
            rlength_array = np.array(oversampled_in[key])
            self.kwargOverSampled[key][offset_in:repeatAmount + offset_in] = rlength_array[:]

    def over_sample_arrays(self, smote=False):
        if smote:
            return self.smote_oversample()
        else:
            return self.minority_oversample_with_noise()

    def minority_oversample_with_noise(self):
        offset = 0
        # pool = mp.Pool()
        for i in range(len(self.kwargShuf['labels'])):
            labelIndex = self.kwargShuf['labels'][i]
            if self.overSampleAmount[labelIndex] < 10:
                std = 0.03
            else:
                std = 0.05
            # pool.apply_async(self.oversample_mp, args=(i, offset, std, labelIndex), callback=self.collect_results)
            self.collect_results(self.oversample_mp(i, offset, std, labelIndex))
            offset += int(self.overSampleAmount[labelIndex])
        # pool.close()
        # pool.join()

        # for i, output in enumerate(outputs):
        #     self.collect_results(output)
        #     print('combining results...', i, len(outputs))

        print("Before Shuffling")
        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_{}'.format(self.randnum),
                                                        **self.kwargOverSampled)
        print("After Shuffling")

        return self.kwargOverSampledShuf

    def smote_oversample(self):
        sm = over_sampling.SMOTE(random_state=42, n_jobs=30)
        images, labels = sm.fit_sample(X=self.kwargShuf['images'], y=self.kwargShuf['labels'])

        self.kwargOverSampledShuf = self.shuffle_arrays(memmapName='oversampled_smote_{}'.format(self.randnum),
                                                        images=images, labels=labels)

        return self.kwargOverSampledShuf


class CreateArrays(object):
    def __init__(self, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts,
                 hostTypes=None, nHostTypes=None):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.minZ = minZ
        self.maxZ = maxZ
        self.numOfRedshifts = numOfRedshifts
        self.ageBinning = AgeBinning(minAge, maxAge, ageBinSize)
        # self.numOfAgeBins = self.ageBinning.age_bin(maxAge - 0.1) + 1
        self.numOfAgeBins = 3
        self.nLabels = nTypes * self.numOfAgeBins * nHostTypes
        self.createLabels = CreateLabels(self.nTypes, self.minAge, self.maxAge, self.ageBinSize, self.typeList,
                                         hostTypes, nHostTypes)
        self.hostTypes = hostTypes

    def combined_sn_gal_templates_to_arrays(self, args):
        snTemplateLocation, snTempList, galTemplateLocation, galTempList, snFractions, ageIndexes = args
        # print(ageIndexes)
        images = np.empty((0, int(self.nw)), np.float16)  # Number of pixels
        labelsIndexes = []
        filenames = []
        typeNames = []

        for j, gal in enumerate(galTempList):
            galFilename = galTemplateLocation + gal if galTemplateLocation is not None else None
            for i, sn in enumerate(snTempList):
                nCols = 1000
                readSpectra = ReadSpectra(self.w0, self.w1, self.nw, snTemplateLocation + sn, galFilename)
                for ageidx in ageIndexes[sn]:
                    if ageidx >= nCols:
                        break
                    for snCoeff in snFractions:
                        galCoeff = 1 - snCoeff
                        if self.numOfRedshifts == 1:
                            redshifts = [self.minZ]
                        else:
                            redshifts = np.random.uniform(low=self.minZ, high=self.maxZ, size=self.numOfRedshifts)
                        # print("redshifts::: ",redshifts)
                        for z in redshifts:
                            # print(z)
                            tempWave, tempFlux, nCols, ages, tType, tMinIndex, tMaxIndex = readSpectra.sn_plus_gal_template(
                                ageidx, snCoeff, galCoeff, z)
                            # print("ages::::: ",ages, "\n\nhhhhh::\n\n",ageidx)
                            if tMinIndex == tMaxIndex or not tempFlux.any():
                                print("NO DATA for {} {} ageIdx:{} z>={}".format(galTempList[j], snTempList[i], ageidx,
                                                                                 z))
                                break
                            #print("ages::::: ",ages, "\n\nhhhhh::\n\n",ageidx)
                            if self.minAge < float(ages[ageidx]) < self.maxAge:
                                if self.hostTypes is None:  # Checks if we are classifying by host as well
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx], host=None)
                                else:
                                    labelIndex, typeName = self.createLabels.label_array(tType, ages[ageidx],
                                                                                         host=galTempList[j])
                                if tMinIndex > (self.nw - 1):
                                    continue
                                nonzeroflux = tempFlux[tMinIndex:tMaxIndex + 1]
                                newflux = (nonzeroflux - min(nonzeroflux)) / (max(nonzeroflux) - min(nonzeroflux))
                                newflux2 = np.concatenate((tempFlux[0:tMinIndex], newflux, tempFlux[tMaxIndex + 1:]))
                                images = np.append(images, np.array([newflux2]), axis=0)
                                labelsIndexes.append(
                                    labelIndex)  # labels = np.append(labels, np.array([label]), axis=0)
                                filenames.append(
                                    "{0}_{1}_{2}_{3}_snCoeff{4}_z{5}".format(snTempList[i], tType, str(ages[ageidx]),
                                                                             galTempList[j], snCoeff, (z)))
                                typeNames.append(typeName)
                #print(snTempList[i], nCols, galTempList[j])

        return images, np.array(labelsIndexes).astype(int), np.array(filenames), np.array(typeNames)

    def collect_results(self, result):
        """Uses apply_async's callback to setup up a separate Queue for each process"""
        imagesPart, labelsPart, filenamesPart, typeNamesPart = result
        self.images.extend(imagesPart)
        self.labelsIndexes.extend(labelsPart)
        self.filenames.extend(filenamesPart)
        self.typeNames.extend(typeNamesPart)

    def combined_sn_gal_arrays_multiprocessing(self, snTemplateLocation, snTempFileList, galTemplateLocation,
                                               galTempFileList):
        # TODO: Maybe do memory mapping for these arrays
        self.images = []
        self.labelsIndexes = []
        self.filenames = []
        self.typeNames = []

        if galTemplateLocation is None or galTempFileList is None:
            galTempList = [None]
            galTemplateLocation = None
            snFractions = [1.0]
        else:
            galTempList = temp_list(galTempFileList)
            snFractions = [0.99, 0.98, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        if isinstance(snTempFileList, dict):
            snTempList = list(snTempFileList.keys())
            ageIndexesDict = snTempFileList
        else:
            snTempList = temp_list(snTempFileList)
            ageIndexesDict = None

        galAndSnTemps = list(itertools.product(galTempList, snTempList))
        argsList = []
        for gal, sn in galAndSnTemps:
            if ageIndexesDict is not None:
                ageIdxDict = {k: ageIndexesDict[k] for k in (sn,)}
            else:
                ageIdxDict = {k: range(0, 1000) for k in (sn,)}
            argsList.append((snTemplateLocation, [sn], galTemplateLocation, [gal], snFractions, ageIdxDict))

        # print(argsList)

        #
        #
        # pool = mp.Pool()
        # results = pool.map_async(self.combined_sn_gal_templates_to_arrays, argsList)
        # pool.close()
        # pool.join()
        #
        # outputs = results.get()

        # for i, output in enumerate(outputs):
        #     self.collect_results(output)
        #     print('combining results...', i, len(outputs))

        #
        #
        #

        for i, output in enumerate(argsList):
            outputs = self.combined_sn_gal_templates_to_arrays(output)
            self.collect_results(outputs)
            print('combining results...', i, len(outputs))


        self.images = np.array(self.images)
        self.labelsIndexes = np.array(self.labelsIndexes)
        self.filenames = np.array(self.filenames)
        self.typeNames = np.array(self.typeNames)

        print("Completed Creating Arrays!")

        return self.images, self.labelsIndexes.astype(np.uint16), self.filenames, self.typeNames
