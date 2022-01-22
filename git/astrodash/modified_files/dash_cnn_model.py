import os

import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, AveragePooling1D, BatchNormalization
import keras_tuner as kt
from focal_loss import SparseCategoricalFocalLoss

from keras.callbacks import EarlyStopping
# hyperparameter optimization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from time import time


class AstroDASH(object):

    def __init__(self, trainImages, trainLabels, testImages, testLabels, N, nLabels, typeNamesList,
                 testTypeNames, snTypes):

        self.trainImages = trainImages
        self.trainLabels = trainLabels
        self.testImages = testImages
        self.testLabels = testLabels
        self.N = N
        self. nLabels = nLabels
        self.typeNamesList = typeNamesList
        self.testTypeNames = testTypeNames
        self.snTypes = snTypes
        self.model = None
        self.history = None
        self.predLabels = None
        self.path = os.path.dirname(os.path.abspath(__file__))

    def reshape_input(self, array):

        array = array.reshape(array.shape[0], array.shape[1], 1)
        return array

    def training_result_per_epoch(self, plot_acc=False, plot_loss=False):

        if plot_acc:

            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

        if plot_loss:

            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    def build_model(self):


        # test = set(self.trainLabels)
        # val, freq = np.unique(self.trainLabels, return_counts=True)
        # # print(val, "\n\n", freq, "\n\n", len(self.trainLabels), sum(freq))
        # weight = freq/sum(freq)
        # print(len(weight))
        # test1 = test - set(val)
        # print(test1, len(test))


        weights= np.full((self.nLabels,), 0.25)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # print(len(weights), self.N, "\n\n", weights, "\n\n",self.nLabels)
        reshaped_trainImages = self.reshape_input(self.trainImages)
        self.model = Sequential()
        self.model.add(Conv1D(32, 5, activation="relu", input_shape=(self.N,1)))
        #self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(AveragePooling1D(pool_size=2))
        self.model.add(Dropout(0.30))
        self.model.add(Conv1D(64, 5, activation="relu"))
        #self.model.add(BatchNormalization())
        # self.model.add(AveragePooling1D(pool_size=2))
        self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Dropout(0.30))
        self.model.add(Flatten())
        self.model.add(Dense(self.nLabels, activation = 'softmax'))
        # self.model.compile(loss = SparseCategoricalFocalLoss(gamma=2, class_weight=weights, from_logits=False),
        #                    optimizer = "adam", metrics = ['accuracy'])
        self.model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam",
                  metrics = ['accuracy'])

        self.model.summary()
        self.history = self.model.fit(reshaped_trainImages, self.trainLabels, validation_split= 0.3,
                            batch_size=150,epochs=10, verbose=0)
        acc = self.model.evaluate(reshaped_trainImages, self.trainLabels)
        print("Loss:", acc[0], " Accuracy:", acc[1])


    def predict_model(self):

        reshaped_testImages = self.reshape_input(self.testImages)
        pred = self.model.predict(reshaped_testImages)
        print(pred, pred.shape)
        self.predLabels = pred.argmax(axis=-1)
        acc = accuracy_score(self.testLabels, self.predLabels)
        print("Accuracy:", acc)

        return self.predLabels, pred



class HyperParameterTuning(object):


    def __init__(self, trainImages, trainLabels, testImages, testLabels, N, nLabels, typeNamesList,
                 testTypeNames, snTypes):

        self.trainImages = trainImages
        self.trainLabels = trainLabels
        self.testImages = testImages
        self.testLabels = testLabels
        self.N = N
        self. nLabels = nLabels
        self.typeNamesList = typeNamesList
        self.testTypeNames = testTypeNames
        self.snTypes = snTypes
        self.model = None
        self.history = None
        self.predLabels = None
        self.path = os.path.dirname(os.path.abspath(__file__))

    def reshape_input(self, array):

        array = array.reshape(array.shape[0], array.shape[1], 1)
        return array


    def model_builder(self, hp):

        model = Sequential()
        model.add(Conv1D(32, 5, activation="relu", input_shape=(1024,1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(64, 5, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(306, activation = 'softmax'))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        opt = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(loss=loss , optimizer=opt,
                           metrics=['acc'])


        return model

    def perform_tuning(self):

        reshaped_trainImages = self.reshape_input(self.trainImages)
        reshaped_testImages = self.reshape_input(self.testImages)

        tuner=kt.Hyperband(self.model_builder,
                                   objective='val_acc',
                                   max_epochs=10,
                                   factor=3)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)

        tuner.search(reshaped_trainImages, self.trainLabels, epochs=50, validation_split=0.2,
                     callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
                The hyperparameter search is complete. The optimal learning rate for the optimizer
                is {best_hps.get('learning_rate')}.
                """)

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(reshaped_trainImages, self.trainLabels, epochs=50, validation_split=0.2)
        val_acc_per_epoch = history.history['val_acc']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(reshaped_trainImages, self.trainLabels, epochs=best_epoch, validation_split=0.2)
        eval_result = hypermodel.evaluate(reshaped_testImages, self.testLabels)
        y_pred = hypermodel.predict(reshaped_testImages)
        y_pred = y_pred.argmax(axis=-1)
        print("[test loss, test accuracy]:", eval_result)
        print("\n\n", y_pred,"\n\n",len(y_pred))
        return y_pred