import os

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix



class model_metrics(object):

    def __init__(self, testLabels, predLabels, pred_proba, N, nLabels, typeNamesList,
                 testTypeNames, snTypes):

        self.testLabels = testLabels
        self.N = N
        self. nLabels = nLabels
        self.typeNamesList = typeNamesList
        self.testTypeNames = testTypeNames
        self.snTypes = snTypes
        self.predLabels = predLabels
        self.pred_proba = pred_proba
        self.path = os.path.dirname(os.path.abspath(__file__))


    def aggregated_confusion_matrix(self, aggregateIndexes):

        testLabelsAggregated = np.digitize(self.testLabels, aggregateIndexes) - 1
        predictedLabelsAggregated = np.digitize(self.predLabels, aggregateIndexes) - 1
        confMatrixAggregated = confusion_matrix(testLabelsAggregated, predictedLabelsAggregated)
        np.set_printoptions(precision=2)
        print(confMatrixAggregated)
        return confMatrixAggregated

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap="Blues",
                              name=""):



        if normalize:
            with np.errstate(divide='ignore'):

                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        cm = np.nan_to_num(cm)


        fig, ax = plt.subplots(figsize=(15,15))
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, annot_kws={"size":15}, fmt=fmt, ax=ax, center=0.5, linewidths=0.5, cbar=False,
                square=True, cmap=cmap)

        ax.set_xlabel("Predicted Labels", fontsize=15)
        ax.set_ylabel("True Labels", fontsize=15)
        ax.set_title(title, fontsize=18)
        ax.xaxis.set_ticklabels(classes, size=15)
        ax.yaxis.set_ticklabels(classes, size=15)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(self.path+f"/images/confusion_matrix_{name}")



    def confusion_matrix(self, aggregated_confusion_matrix=False):

        nBins = len(self.typeNamesList)
        confMatrix = pd.DataFrame(confusion_matrix(self.testLabels, self.predLabels,
                                                   labels=list(np.arange(self.nLabels))))


        # Aggregate age conf matrix

        if aggregated_confusion_matrix:
            aggregateAgesIndexes = np.arange(0, nBins + 1, int(nBins / len(self.snTypes)))
            confMatrixAggregateAges = self.aggregated_confusion_matrix(aggregateAgesIndexes)
            classnames = np.copy(self.snTypes)
            if confMatrixAggregateAges.shape[0] < len(classnames):
                classnames = classnames[:-1]
            self.plot_confusion_matrix(confMatrixAggregateAges, classes=classnames, normalize=True,
                                   name='aggregate_ages')

            # Aggregate age and subtypes conf matrix
            aggregateSubtypesIndexes = np.array([0, 18, 30, 39, 51, 54, 57, 60])
            broadTypes = ['Ia', 'Ib', 'Ic', 'II', 'TDE', 'KN', 'SLSNe']
            confMatrixAggregateSubtypes = self.aggregated_confusion_matrix(aggregateSubtypesIndexes)
            self.plot_confusion_matrix(confMatrixAggregateSubtypes, classes=broadTypes, normalize=True,
                                   name='aggregate_subtypes')



        # ACTUAL ACCURACY, broadTYPE ACCURACY, AGE ACCURACY
        typeAndAgeCorrect = 0
        typeCorrect = 0
        broadTypeCorrect = 0
        broadTypeAndAgeCorrect = 0
        typeAndNearAgeCorrect = 0
        broadTypeAndNearAgeCorrect = 0
        for i in range(len(self.testTypeNames)):
            predictedIndex = np.argmax(self.pred_proba[i])

            classification = self.testTypeNames[i].split(': ')
            if len(classification) == 2:
                testType, testAge = classification
            else:
                testGalType, testType, testAge = classification
            actual = self.typeNamesList[predictedIndex].split(': ')
            # actual = self.typeNamesList[self.predLabels]

            if len(actual) == 2:
                actualType, actualAge = actual
            else:
                actualGalType, actualType, actualAge = actual

            testBroadType = testType[0:2]
            actualBroadType = actualType[0:2]
            if testType[0:3] == 'IIb':
                testBroadType = 'Ib'
            if actualType[0:3] == 'IIb':
                actualBroadType = 'Ib'
            nearTestAge = testAge.split(' to ')

            if self.testTypeNames[i] == self.typeNamesList[predictedIndex]:
                typeAndAgeCorrect += 1
            if testType == actualType:  # correct type
                typeCorrect += 1
                if (nearTestAge[0] in actualAge) or (
                        nearTestAge[1] in actualAge):  # check if the age is in the neigbouring bin
                    typeAndNearAgeCorrect += 1  # all correct except nearby bin
            if testBroadType == actualBroadType:  # correct broadtype
                broadTypeCorrect += 1
                if testAge == actualAge:
                    broadTypeAndAgeCorrect += 1
                if (nearTestAge[0] in actualAge) or (
                        nearTestAge[1] in actualAge):  # check if the age is in the neigbouring bin
                    broadTypeAndNearAgeCorrect += 1  # Broadtype and nearby bin

        typeAndAgeAccuracy = float(typeAndAgeCorrect) / len(self.testTypeNames)
        typeAccuracy = float(typeCorrect) / len(self.testTypeNames)
        broadTypeAccuracy = float(broadTypeCorrect) / len(self.testTypeNames)
        broadTypeAndAgeAccuracy = float(broadTypeAndAgeCorrect) / len(self.testTypeNames)
        typeAndNearAgeAccuracy = float(typeAndNearAgeCorrect) / len(self.testTypeNames)
        broadTypeAndNearAgeAccuracy = float(broadTypeAndNearAgeCorrect) / len(self.testTypeNames)

        print("typeAndAgeAccuracy : " + str(typeAndAgeAccuracy))
        print("typeAccuracy : " + str(typeAccuracy))
        print("broadTypeAccuracy : " + str(broadTypeAccuracy))
        print("broadTypeAndAgeAccuracy: " + str(broadTypeAndAgeAccuracy))
        print("typeAndNearAgeAccuracy : " + str(typeAndNearAgeAccuracy))
        print("broadTypeAndNearAgeAccuracy : " + str(broadTypeAndNearAgeAccuracy))



