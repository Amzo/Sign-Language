import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

from lib.debug import LogInfo


class customModel(tf.keras.callbacks.Callback):
    def __init__(self, debug=None):
        self.imgFeatures = None
        self.debug = debug

        self.createModel()

    def on_epoch_end(self, epoch, logs=None):
        try:
            f1score = round(
                2 * ((logs['val_precision'] * logs['val_recall']) / (logs['val_precision'] + logs['val_recall'])), 4)
        except ZeroDivisionError:
            f1score = 0

        if self.debug.get():
            self.debugWindow.logText(self.debug.LogInfo.info.value,
                                     "End epoch {}; loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {}"
                                     .format(epoch, round(logs['loss'], 4), round(logs['accuracy'], 4),
                                             round(logs['val_loss'], 4), round(logs['val_accuracy'], 4), ))

        self.precision.set(str(round(logs['val_precision'], 4)))
        self.recall.set(str(round(logs['val_recall'], 4)))
        self.f1.set(str(f1score))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def fitKnn(self):
        signLanguageSet = pd.read_csv("C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\Train\\data.csv",
                                      names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                             "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                             "40", "41", "42", "Key"])

        signTestSet = pd.read_csv("C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\Test\\data.csv",
                                  names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                         "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                         "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                         "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                         "40", "41", "42", "Key"])

        signLanguageSet.drop('Image', inplace=True, axis=1)
        signTestSet.drop('Image', inplace=True, axis=1)

        x_test = signTestSet.iloc[:, :42].values
        y_test = signTestSet['Key'].values

        x = signLanguageSet.iloc[:, :42].values
        y = signLanguageSet['Key'].values

        neigh = KNeighborsClassifier(n_neighbors=3)

        print(x.shape)
        neigh.fit(x, y)

        y_pred = neigh.predict(x_test)

        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=1))
        print(confusion_matrix(y_test, y_pred))

        knnPickle = open('C:\\Users\\Amzo\\Documents\\Robotics Assignment\\NewData\\Model\\knn.pkl', 'wb')
        pickle.dump(neigh, knnPickle)

    def makePrediction(self, checkFrame, fingerPoints):

        loadedLabels = pickle.load(open('C:\\Users\\Amzo\\PycharmProjects\\RoboticsAssignment\\classes.pkl', 'rb'))

        results = self.model.predict(
            [np.expand_dims(checkFrame, axis=0), np.asarray(fingerPoints).reshape(1, -1)])

        print(str(loadedLabels.inverse_transform(results)))

        # Instead of getting a prediction every frame: 24 frames per second
        # Find out most frequent prediction in those 24 frames and display it every
        # 1 second
        # if len(self.predictionList) >= 24:
        #   print(mode(self.predictionList))
        #  self.predictionList == []

    def train(self, log_window=None, model_save_dir=None, data_dir=None, csv_file=None):
        self.loadTrainData(log_window=log_window, data_dir=data_dir, csv_file=csv_file)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=data_dir + "/models/",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if log_window.debug.get():
            log_window.debugWindow.logText(LogInfo.info.debug, "Creating Model")

        if log_window.debug.get():
            log_window.debugWindow.logText(LogInfo.debug.value, "Compiling the model loss: categorical crossentropy, optimizer: rmsprop")
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                           metrics=['categorical_crossentropy', 'accuracy', tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall()])

        if log_window.debug.get():
            log_window.debugWindow.logText(LogInfo.debug.value, "Splitting data 33% of data for testing")
        x1_train, x1_test, y_train, y_test = train_test_split(self.imgFeatures,
                                                              self.encoded_Y, test_size=0.33,
                                                              random_state=42)

        if log_window.debug.get():
            log_window.debugWindow.logText(LogInfo.debug.value, "Augmenting the data and training")

        # gen_flow = ourModels.augmentData(x1_train, x2_train, y_train)
        self.model.fit(x1_train, y_test, verbose=1, validation_data=(x1_test, y_test),
                       steps_per_epoch=len(x1_train) / 32, epochs=self.epochs,
                       callbacks=[self, model_checkpoint_callback], )

        score = self.model.evaluate([self.testImages, self.testFeatures], self.encoded_test_Y)
        print(score)

    def createModel(self):
        img = tf.keras.Input(shape=(100, 100, 3))
        features = tf.keras.Input(shape=(42,))

        encoded = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(img)
        encoded = tf.keras.layers.Dense(64, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=tf.keras.regularizers.l2(1e-4),
                                        activity_regularizer=tf.keras.regularizers.l2(1e-5))(encoded)

        encoded = tf.keras.layers.Dropout(0.5)(encoded)
        encoded = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(encoded)

        encoded = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(encoded)
        encoded = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(encoded)

        x = tf.keras.layers.Flatten()(encoded)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        main_output = tf.keras.layers.Dense(24, activation="softmax", name='main_output')(x)

        self.model = tf.keras.Model([img], [main_output])

    def loadTrainData(self, log_window=None, data_dir=None, csv_file=None):
        if not os.path.isdir(data_dir + "/Train/"):
            log_window.logText(LogInfo.error.value, "Couldn't find train dir in {}".format(self.saveLocation.get()))
        elif not os.path.isdir(data_dir + "/Test/"):
            log_window.debugWindow.logText(LogInfo.error.value, "Couldn't find test dir in {}".format(self.saveLocation.get()))
        else:
            log_window.logText(LogInfo.info.value, "Calculating number of images")

            log_window.logText(LogInfo.info.value, "Loading {} image files".format(self.dataCount.get()))

            # find the increment value of the progress bar
            updateValue = 100.00 / float(self.dataCount.get())

            signLanguageSet = read_csv(csv_file,
                                       names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                              "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                              "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                              "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                              "40", "41", "42", "Key"])

            signTestSet = read_csv(data_dir + "/Test/data.csv",
                                   names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                          "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                          "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                          "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                          "40", "41", "42", "Key"])
            train_image = []
            train_label = []
            test_images = []
            test_labels = []

            self.logText(LogInfo.info.value, "Finished reading cvs")

            for i in range(signLanguageSet.shape[0]):
                img = cv2.imread(self.saveLocation.get() + "/Train/" + signLanguageSet['Image'][i])

                assert img.shape == (100, 100, 3)

                label = signLanguageSet['Key'][i]

                img = img / 255
                train_image.append(np.asarray(img))
                train_label.append(label)

                self.progressValue.set(i * updateValue)
                k = i

            for i in range(signTestSet.shape[0]):
                img = cv2.imread(self.saveLocation.get() + "/Test/" + signTestSet['Image'][i])

                assert img.shape == (100, 100, 3)

                label = signTestSet['Key'][i]
                img = img / 255
                test_images.append(np.asarray(img))
                test_labels.append(label)
                self.progressValue.set(k * updateValue)

            signLanguageSet.drop('Image', inplace=True, axis=1)
            signLanguageSet.drop('Key', inplace=True, axis=1)

            signTestSet.drop('Image', inplace=True, axis=1)
            signTestSet.drop('Key', inplace=True, axis=1)

            self.imgFeatures = np.array(train_image)
            labels = np.array(train_label)
            self.signFeatures = np.array(signLanguageSet)

            self.testImages = np.array(test_images)
            testLabels = np.array(test_labels)
            self.testFeatures = np.array(signTestSet)

            encoder = LabelBinarizer()
            encoder.fit_transform(labels)
            self.encoded_Y = encoder.transform(labels)
            self.encoded_test_Y = encoder.transform(testLabels)

            output = open('classes.pkl', 'wb')
            pickle.dump(encoder, output)

            self.logText(LogInfo.info.value, "Finished loading images")

            self.train()
