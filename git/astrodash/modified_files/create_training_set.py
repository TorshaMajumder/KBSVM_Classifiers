import pickle
import numpy as np
import zipfile
import gzip
import os
import random
import copy
from collections import OrderedDict
from create_arrays import AgeBinning, CreateLabels, ArrayTools, CreateArrays
from helpers import temp_list
import pandas as pd

from sklearn.model_selection import train_test_split

class CreateTrainingSet(object):

    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList,
                 minZ, maxZ, numOfRedshifts, galTemplateLocation, galTempFileList, hostTypes, nHostTypes,
                 trainFraction):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidTempFileList = snidTempFileList
        self.galTemplateLocation = galTemplateLocation
        self.galTempFileList = galTempFileList
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.trainFraction = trainFraction
        self.ageBinning = AgeBinning(self.minAge, self.maxAge, self.ageBinSize)
        # self.numOfAgeBins = self.ageBinning.age_bin(self.maxAge - 0.1) + 1
        self.numOfAgeBins = 3
        self.nLabels = self.nTypes * self.numOfAgeBins * nHostTypes
        print(self.nLabels)
        self.createArrays = CreateArrays(w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList, minZ, maxZ,
                                         numOfRedshifts, hostTypes, nHostTypes)
        self.arrayTools = ArrayTools(self.nLabels, self.nw)

    def type_amounts(self, labels):
        counts = self.arrayTools.count_labels(labels)

        return counts

    def all_templates_to_arrays(self, snTempFileList, galTemplateLocation):
        """
        Parameters
        ----------
        snTempFileList : list or dictionary
        galTemplateLocation

        Returns
        -------
        """
        images, labels, filenames, typeNames = self.createArrays.combined_sn_gal_arrays_multiprocessing(
            self.snidTemplateLocation, snTempFileList, galTemplateLocation, self.galTempFileList)

        arraysShuf = self.arrayTools.shuffle_arrays(images=images, labels=labels, filenames=filenames,
                                                    typeNames=typeNames, memmapName='all')

        typeAmounts = self.type_amounts(labels)

        return arraysShuf, typeAmounts

    def train_test_split(self, arrays):
        """
        Split training set before creating arrays.
        Maybe should change this to include ages in train/test split instead of just SN files.
        """
        X = pd.DataFrame(arrays['images'])
        y = copy.copy(arrays)
        y.pop('images', None)
        y = pd.DataFrame(y)
        train_X, test_X , train_y, test_y = train_test_split(X, y, train_size=self.trainFraction, random_state=42)

        return train_X, test_X , train_y, test_y




    def sort_data(self):

        arrays, typeAmounts = self.all_templates_to_arrays(self.snidTempFileList, self.galTemplateLocation)
        # print(len(arrays))
        train_X, test_X , train_y, test_y = self. train_test_split(arrays)

        #
        # Total Number of labels unused in the test set
        #


        # idx = [typeNameList[i] for i in label_diff]
        # print(idx)





        trainImages, trainLabels, trainFilenames, trainTypeNames = train_X.to_numpy(), \
                                                                   train_y['labels'].to_numpy(), \
                                                                    train_y['filenames'].to_numpy(), \
                                                                   train_y['typeNames'].to_numpy()

        testImages, testLabels, testFilenames, testTypeNames = test_X.to_numpy(), \
                                                               test_y['labels'].to_numpy(), \
                                                                test_y['filenames'].to_numpy(), \
                                                               test_y['typeNames'].to_numpy()

        typeAmounts = self.type_amounts(trainLabels)


        train_images = np.array(trainImages).tolist()
        train_labels = np.array(trainLabels).tolist()
        test_images = np.array(testImages).tolist()
        test_labels = np.array(testLabels).tolist()
        test_TypeNames = np.array(testTypeNames).tolist()
        train_TypeNames = np.array(trainTypeNames).tolist()
        train_Filenames = np.array(trainFilenames).tolist()
        test_Filenames = np.array(testFilenames).tolist()



        #
        # Total Number of labels unused in the test set
        #
        label_diff = list(set(test_labels) - set(train_labels))
        # print(label_diff, len(label_diff))
        # idx = [typeNameList[i] for i in label_diff]
        # print(idx)
        if label_diff:
            idx = [i for i, v in enumerate(test_labels) if v in label_diff]
            # print(idx)

            images = [test_images[val] for i, val in enumerate(idx)]
            labels = [test_labels[val] for i, val in enumerate(idx)]
            label_names = [test_TypeNames[val] for i, val in enumerate(idx)]
            file_names = [test_Filenames[val] for i, val in enumerate(idx)]
            # print(images, labels)
            train_labels.extend(labels)
            train_images.extend(images)
            train_TypeNames.extend(label_names)
            train_Filenames.extend(file_names)
            train_images = np.asarray(train_images)
            train_labels = np.asarray(train_labels)
            train_TypeNames = np.asarray(train_TypeNames)
            train_Filenames = np.asarray(train_Filenames)

            # print(train_images.shape, train_labels.shape, train_TypeNames.shape, train_Filenames.shape)

            return ((train_images, train_labels, train_Filenames, train_TypeNames),
                    (testImages, testLabels, testFilenames, testTypeNames),
                    typeAmounts)



        return ((trainImages, trainLabels, trainFilenames, trainTypeNames),
                (testImages, testLabels, testFilenames, testTypeNames),
                typeAmounts)


class SaveTrainingSet(object):
    def __init__(self, snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge, ageBinSize, typeList,
                 minZ, maxZ, numOfRedshifts, galTemplateLocation=None, galTempFileList=None, hostTypes=None,
                 nHostTypes=1, trainFraction=0.8):
        self.snidTemplateLocation = snidTemplateLocation
        self.snidTempFileList = snidTempFileList
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.nTypes = nTypes
        self.minAge = minAge
        self.maxAge = maxAge
        self.ageBinSize = ageBinSize
        self.typeList = typeList
        self.createLabels = CreateLabels(nTypes, minAge, maxAge, ageBinSize, typeList, hostTypes, nHostTypes)

        self.createTrainingSet = CreateTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge,
                                                   maxAge, ageBinSize, typeList, minZ, maxZ, numOfRedshifts,
                                                   galTemplateLocation, galTempFileList, hostTypes, nHostTypes,
                                                   trainFraction)
        self.sortData = self.createTrainingSet.sort_data()
        # print(len(self.sortData))
        self.trainImages = self.sortData[0][0]
        self.trainLabels = self.sortData[0][1]
        self.trainFilenames = self.sortData[0][2]
        self.trainTypeNames = self.sortData[0][3]
        self.testImages = self.sortData[1][0]
        self.testLabels = self.sortData[1][1]
        self.testFilenames = self.sortData[1][2]
        self.testTypeNames = self.sortData[1][3]
        self.typeAmounts = self.sortData[2]

        self.typeNamesList = self.createLabels.type_names_list()

    def type_amounts(self):
        for i in range(len(self.typeNamesList)):
            print(str(self.typeAmounts[i]) + ": " + str(self.typeNamesList[i]))
        return self.typeNamesList, self.typeAmounts

    def save_arrays(self, saveFilename):
        arraysToSave = {'trainImages.npy.gz': self.trainImages, 'trainLabels.npy.gz': self.trainLabels,
                        'testImages.npy.gz': self.testImages, 'testLabels.npy.gz': self.testLabels,
                        'testTypeNames.npy.gz': self.testTypeNames, 'typeNamesList.npy.gz': self.typeNamesList,
                        'trainFilenames.npy.gz': self.trainFilenames, 'trainTypeNames.npy.gz': self.trainTypeNames}

        try:
            print("SIZE OF ARRAYS TRAINING:")
            print(self.trainImages.nbytes, self.testImages.nbytes)
            print(self.trainLabels.nbytes, self.testLabels.nbytes)
            print(self.trainFilenames.nbytes, self.testFilenames.nbytes)
            print(self.trainTypeNames.nbytes, self.testTypeNames.nbytes)
        except Exception as e:
            print(f"Exception Raised --- {e}")

        for filename, array in arraysToSave.items():
            f = gzip.GzipFile(filename, "w")
            np.save(file=f, arr=array)
            f.close()

        with zipfile.ZipFile(saveFilename, 'w') as myzip:
            for f in arraysToSave.keys():
                myzip.write(f)

        print("Saved dataset to: " + saveFilename)

        # Delete npy.gz files
        for filename in arraysToSave.keys():
            os.remove(filename)


def create_training_set_files(dataDirName, minZ=0, maxZ=0, numOfRedshifts=80, trainWithHost=True, classifyHost=False,
                              trainFraction=0.8):



    with open(os.path.join(dataDirName, 'training_params.pickle'), 'rb') as f1:
        pars = pickle.load(f1)
    nTypes, w0, w1, nw, minAge, maxAge, ageBinSize, typeList = pars['nTypes'], pars['w0'], pars['w1'], \
                                                               pars['nw'], pars['minAge'], pars['maxAge'], \
                                                               pars['ageBinSize'], pars['typeList']

    #
    # default value is None/1
    #
    hostList, nHostTypes = None, 1

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))

    snidTemplateLocation = os.path.join(scriptDirectory, "templates/training_set/")
    snidTempFileList = snidTemplateLocation + 'templist.txt'


    #
    # trainWithHost == True
    #
    if trainWithHost:
        galTemplateLocation = os.path.join(scriptDirectory, "templates/superfit_templates/gal/")
        galTempFileList = galTemplateLocation + 'gal.list'

        #
        # classifyHost == False
        #
        if classifyHost:
            hostList = pars['galTypeList']
            nHostTypes = len(hostList)
    else:

        #
        # ??
        #
        galTemplateLocation, galTempFileList = None, None


    #
    #
    #
    saveTrainingSet = SaveTrainingSet(snidTemplateLocation, snidTempFileList, w0, w1, nw, nTypes, minAge, maxAge,
                                      ageBinSize, typeList, minZ, maxZ, numOfRedshifts, galTemplateLocation,
                                      galTempFileList, hostList, nHostTypes, trainFraction)
    typeNamesList, typeAmounts = saveTrainingSet.type_amounts()

    saveFilename = os.path.join(dataDirName, 'dataset.zip')
    saveTrainingSet.save_arrays(saveFilename)

    return saveFilename


if __name__ == '__main__':
    trainingSetFilename = create_training_set_files('data_files/', minZ=0, maxZ=0, numOfRedshifts=80,
                                                    trainWithHost=False, classifyHost=False, trainFraction=0.8)

