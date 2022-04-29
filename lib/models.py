import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from tensorflow.python.keras.layers import concatenate

from lib.debug import LogInfo


def get_model_name(k):
    return 'model_' + str(k) + '.h5'


class customModel(tf.keras.callbacks.Callback):
    def __init__(self, root_window=None):
        self.encoder = None
        self.results = None
        self.prevF1Score = 0
        self.Fold_Test_Input2 = None
        self.Fold_Test_OutPut = None
        self.Fold_Test_Input1 = None
        self.Fold_Train_OutPut = None
        self.Fold_Train_Input2 = None
        self.Fold_Train_Input1 = None
        self.fold_var = 1
        self.x2Test = None
        self.x1Test = None
        self.yTest = None
        self.yValidate = None
        self.yTrain = None
        self.x2Validate = None
        self.x1Validate = None
        self.x2Train = None
        self.x1Train = None
        self.testLabel = None
        self.validateLabel = None
        self.trainLabel = None
        self.signLanguageSet = None
        self.modelFeatures = None
        self.model = None
        self.signFeatures = None
        self.testImages = None
        self.testFeatures = None
        self.encoded_test_Y = None
        self.encoded_Y = None
        self.imgFeatures = None
        self.train_image = []
        self.train_label = []
        self.trainIndices = []

        self.validate_image = []
        self.validate_label = []
        self.validateIndices = []

        self.test_image = []
        self.test_label = []
        self.testIndices = []
        self.rootWindow = root_window

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        updateValue = 100.00 / int(self.rootWindow.trainTab.epochInput.get())

        self.rootWindow.trainTab.logText(LogInfo.info.value,
                                         "End epoch {}; loss: {} - accuracy: {} - val_loss: {} - val_accuracy: {}"
                                         .format(epoch, round(logs['loss'], 4), round(logs['accuracy'], 4),
                                                 round(logs['val_loss'], 4), round(logs['val_accuracy'], 4), ))

        # epoch starts from 0, add + 1
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Updating progress bar by (epoch {} * {})".format(epoch, updateValue))

        self.rootWindow.trainTab.progressValue.set(epoch * updateValue)

    def fitKnn(self):
        self.rootWindow.trainTab.logText(LogInfo.info.value,
                                         "Reading {}".format(self.rootWindow.trainTab.csvFile.get()))
        signLanguageSet = pd.read_csv(self.rootWindow.trainTab.csvFile.get(),
                                      names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                             "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                             "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                             "40", "41", "42", "Key"])

        signLanguageSet = shuffle(signLanguageSet)

        signLanguageSet.drop('Image', inplace=True, axis=1)

        signLanguageSet['split'] = np.random.randn(signLanguageSet.shape[0], 1)

        msk = np.random.rand(len(signLanguageSet)) <= 0.7

        train = signLanguageSet[msk]
        test = signLanguageSet[~msk]

        xTrain = train.iloc[:, :42].values
        yTrain = train['Key'].values

        xTest = test.iloc[:, :42].values
        yTest = test['Key'].values

        neigh = KNeighborsClassifier(n_neighbors=int(self.rootWindow.trainTab.neighInput.get()))

        neigh.fit(xTrain, yTrain)

        yPred = neigh.predict(xTest)

        self.rootWindow.trainTab.logText(LogInfo.info.value,
                                         "KNN Accuracy on test data " + str(accuracy_score(yTest, yPred)))

        self.rootWindow.trainTab.logText(LogInfo.info.value, classification_report(yTest, yPred, zero_division=1))
        self.rootWindow.trainTab.logText(LogInfo.info.value, "Confusion Matrix")
        self.rootWindow.trainTab.logText(10, str(confusion_matrix(yTest, yPred)))

        precision = round(precision_score(yTest, yPred, average='macro'), 2)
        recall = round(recall_score(yTest, yPred, average='macro'), 2)
        f1Score = round(2 * ((precision * recall) / (precision + recall)), 4)

        self.rootWindow.trainTab.precision.set(precision)
        self.rootWindow.trainTab.recall.set(recall)
        self.rootWindow.trainTab.f1.set(f1Score)

        savePath = os.path.split(str(self.rootWindow.trainTab.csvFile.get()))

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Saving KNN to {}/knn.pkl".format(savePath[0]))

        knnPickle = open(savePath[0] + "/knn.pkl", 'wb')
        pickle.dump(neigh, knnPickle)

    def makePrediction(self, points=None, check_frame=None):
        if self.rootWindow.predictTab.selectedModel.get() == "KNN":
            self.results = self.rootWindow.predictTab.loaded_model.predict(points)
        elif self.rootWindow.predictTab.selectedModel.get() == "CNN":
            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Getting CNN Prediction")

            cv2.imshow("test", check_frame)
            cv2.waitKey(1)
            predFrame = np.expand_dims(check_frame, axis=0)
            prediction = self.model.predict([predFrame, points])
            self.results = self.encoder.inverse_transform(prediction)

    def appendImage(self, dirs=None, index=None, img_file=None):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

        assert img.shape == (100, 100, 3)

        if dirs == "Train":
            self.trainLabel = self.signLanguageSet['Key'][index]

            img = img / 255
            self.train_image.append(np.asarray(img))
            self.train_label.append(self.trainLabel)

            self.trainIndices.append(int(index))

        elif dirs == "Validate":
            self.validateLabel = self.signLanguageSet['Key'][index]

            img = img / 255
            self.validate_image.append(np.asarray(img))
            self.validate_label.append(self.validateLabel)

            self.validateIndices.append(int(index))

        elif dirs == "Test":
            self.testLabel = self.signLanguageSet['Key'][index]

            img = img / 255
            self.test_image.append(np.asarray(img))
            self.test_label.append(self.testLabel)

            self.testIndices.append(int(index))

    def gen_flow_for_two_inputs(self, gen):
        genX1 = gen.flow(self.Fold_Train_Input1, self.Fold_Train_OutPut, batch_size=16, seed=666)
        genX2 = gen.flow(self.Fold_Train_Input1, self.Fold_Train_Input2, batch_size=16, seed=666)

        while True:
            X1i = genX1.next()
            X2i = genX2.next()

            # Assert arrays are equal - this was for peace of mind, but slows down training
            np.testing.assert_array_equal(X1i[0], X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

    def train(self):
        self.signLanguageSet = pd.read_csv(self.rootWindow.trainTab.saveLocation.get() + "/data.csv",
                                           names=["Image", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                                  "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                                                  "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                                                  "40", "41", "42", "Key"])

        updateValue = 100 / len(self.signLanguageSet)

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Augmenting the data and training")

        # This code was to work around a csv file with non-existence entries which built up over time with data collection
        # Check to see if the image name exists with in the specified dir matching the key
        # csv file has since been cleaned up but code has remained.
        for i in range(self.signLanguageSet.shape[0]):
            if os.path.exists(
                    self.rootWindow.trainTab.saveLocation.get() + "/Train/" + self.signLanguageSet['Key'][i] + "/" +
                    self.signLanguageSet['Image'][
                        i]):

                img = self.rootWindow.trainTab.saveLocation.get() + "/Train/" + self.signLanguageSet['Key'][i] + "/" + \
                      self.signLanguageSet['Image'][i]

                self.appendImage(dirs="Train", index=i, img_file=img)
            elif os.path.exists(
                    self.rootWindow.trainTab.saveLocation.get() + "/Validate/" + self.signLanguageSet['Key'][i] + "/" +
                    self.signLanguageSet['Image'][
                        i]):

                img = self.rootWindow.trainTab.saveLocation.get() + "/Validate/" + self.signLanguageSet['Key'][
                    i] + "/" + \
                      self.signLanguageSet['Image'][i]

                self.appendImage(dirs="Validate", index=i, img_file=img)

            elif os.path.exists(
                    self.rootWindow.trainTab.saveLocation.get() + "/Test/" + self.signLanguageSet['Key'][i] + "/" +
                    self.signLanguageSet['Image'][
                        i]):

                img = self.rootWindow.trainTab.saveLocation.get() + "/Test/" + self.signLanguageSet['Key'][
                    i] + "/" + \
                      self.signLanguageSet['Image'][i]

                self.appendImage(dirs="Test", index=i, img_file=img)

            self.rootWindow.trainTab.progressValue.set(i + 1 * updateValue)

        # drop none needed columns and create the features and labels for training
        self.signLanguageSet.drop('Image', inplace=True, axis=1)
        self.signLanguageSet.drop('Key', inplace=True, axis=1)
        featureList = self.signLanguageSet.iloc[self.trainIndices, :]
        validateFeatureList = self.signLanguageSet.iloc[self.validateIndices, :]
        testFeatureList = self.signLanguageSet.iloc[self.testIndices, :]

        self.x1Train = np.array(self.train_image)
        trainLabels = np.array(self.train_label)
        self.x2Train = np.array(featureList)

        self.x1Validate = np.array(self.validate_image)
        validateLabels = np.array(self.validate_label)
        self.x2Validate = np.array(validateFeatureList)

        self.x1Test = np.array(self.test_image)
        testLabels = np.array(self.test_label)
        self.x2Test = np.array(testFeatureList)

        encoder = LabelBinarizer()
        encoder.fit(trainLabels)
        self.yTrain = encoder.transform(trainLabels)
        self.yValidate = encoder.transform(validateLabels)
        self.yTest = encoder.transform(testLabels)

        output = open(self.rootWindow.trainTab.modelSaveLocation.get() + '/classes.pkl', 'wb')
        pickle.dump(encoder, output)
        output.close()

        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} training images".format(len(self.x1Train)))
        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} validation images".format(len(self.x1Validate)))
        self.rootWindow.trainTab.logText(LogInfo.info.value, "Found {} testing images".format(len(self.x1Test)))

        gen = ImageDataGenerator(
            rescale=1. / 255.0,
            rotation_range=2,
            zoom_range=0.3,
            brightness_range=[0.8, 1.3],
            horizontal_flip=True,
            vertical_flip=True)

        gen_flow = self.gen_flow_for_two_inputs(gen)

        # prepare to use kfold on multiple inputs
        kfold = KFold(n_splits=3, shuffle=True)

        inputID = []
        outputID = []

        # since we're using kfold, join the validation data to train data before splitting
        # keep test data model evaluation
        np.concatenate((self.x1Train, self.x1Validate), axis=0)
        np.concatenate((self.x2Train, self.x2Validate), axis=0)
        np.concatenate((self.yTrain, self.yValidate), axis=0)

        for x in range(len(self.x1Train)):
            inputID.append(self.x1Train[x])
            outputID.append(self.yTrain[x])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.rootWindow.trainTab.modelSaveLocation.get())

        for trainID, testID in kfold.split(inputID, outputID):
            # Call checkpoint each iteration to update the model save name
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                self.rootWindow.trainTab.modelSaveLocation.get() + '/' + get_model_name(self.fold_var),
                monitor='val_accuracy', verbose=1,
                save_best_only=True, mode='max')

            self.Fold_Train_Input1, self.Fold_Train_Input2 = self.x1Train[trainID], self.x2Train[trainID]
            self.Fold_Train_OutPut = self.yTrain[trainID]

            self.Fold_Test_Input1, self.Fold_Test_Input2 = self.x1Train[testID], self.x2Train[testID]
            self.Fold_Test_OutPut = self.yTrain[testID]

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

            if self.fold_var == 1:
                # Since the weights are loaded after every kfold has been trained, only set the optimizer once
                self.createModel()
                opt = tf.keras.optimizers.Adam(epsilon=1e-8, beta_1=0.9, beta_2=0.999)

                if self.rootWindow.debug.get():
                    self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                        "Compiling the model loss: categorical crossentropy, optimizer: Adam")

                self.model.compile(loss='categorical_crossentropy', optimizer=opt,
                                   metrics=['categorical_crossentropy', 'accuracy', tf.keras.metrics.Precision(),
                                            tf.keras.metrics.Recall()])

                print(self.model.summary())
            else:
                # Load best previous weights
                self.model.load_weights(
                    self.rootWindow.trainTab.modelSaveLocation.get() + "/model_" + str(self.fold_var - 1) + ".h5")

            self.rootWindow.trainTab.logText(LogInfo.info.value,
                                             "Starting training on fold number {}".format(self.fold_var))
            self.model.fit(gen_flow, verbose=1,
                           epochs=self.rootWindow.trainTab.epochs,
                           validation_data=(
                               [self.Fold_Test_Input1, self.Fold_Test_Input2], self.Fold_Test_OutPut),
                           steps_per_epoch=len(self.Fold_Train_Input1) / 16,
                           callbacks=[self, model_checkpoint_callback, tensorboard_callback, reduce_lr], )

            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Loading best saved model")

            self.model = tf.keras.models.load_model(
                self.rootWindow.trainTab.modelSaveLocation.get() + "/model_" + str(self.fold_var) + ".h5")

            if self.rootWindow.debug.get():
                self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Evaluating the best model on test data")

            score = self.model.evaluate([self.x1Test, self.x2Test], self.yTest)

            # Score is a list containing loss, categorical_crossentropy, accuracy, precision, recall

            precision = score[3]
            recall = score[4]
            f1Score = round(2 * ((precision * recall) / (precision + recall)), 4)

            if f1Score > self.prevF1Score:
                self.model.save(self.rootWindow.trainTab.modelLocation.get() + "/bestModel")

                self.rootWindow.trainTab.precision.set(round(precision, 2))
                self.rootWindow.trainTab.recall.set(round(recall, 2))
                self.rootWindow.trainTab.f1.set(round(f1Score, 2))

            self.rootWindow.trainTab.logText(LogInfo.info.value, "Finish train Fold number {} ".format(self.fold_var))
            self.fold_var += 1
            self.prevF1Score = f1Score

    def loadModel(self):
        self.encoder = pickle.load(open(self.rootWindow.predictTab.modelLocation.get() + "/../classes.pkl", 'rb'))
        self.model = tf.keras.models.load_model(self.rootWindow.predictTab.modelLocation.get())

    def createModel(self):
        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Creating a model for fingerPoints")

        fingerPoints = tf.keras.Input(shape=(42,), name='FingerPointInput')
        x = tf.keras.layers.Dense(42, activation="relu")(fingerPoints)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.Model(inputs=fingerPoints, outputs=x)

        img = (100, 100, 3)

        inputs = tf.keras.Input(shape=img, name='ImageInput')

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value, "Loading xception model")

        xceptionModel = tf.keras.applications.xception.Xception(input_tensor=inputs, include_top=False,
                                                                weights='imagenet')

        for layer in xceptionModel.layers[:-2]:
            layer.trainable = False

        y = tf.keras.layers.Dense(512, activation="relu")(xceptionModel.output, training=True)
        y = tf.keras.layers.Flatten()(y)
        y = tf.keras.layers.Dense(256, activation="relu")(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Dense(128, activation="relu")(y)
        y = tf.keras.Model(xceptionModel.input, y)

        if self.rootWindow.debug.get():
            self.rootWindow.debugWindow.logText(LogInfo.debug.value,
                                                "Concatenating outputs of fingerpoint model and xception model")
        combined = concatenate([x.output, y.output])

        combined = tf.keras.layers.Dense(64, activation="relu")(combined)
        z = tf.keras.layers.Dense(24, activation="softmax")(combined)

        self.model = tf.keras.Model(inputs=[y.input, x.input], outputs=z)
