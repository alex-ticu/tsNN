import tensorflow as tf
import numpy as np


class KerasImagePreprocessor():

    def __init__(self, dataset, **imageDataGeneratorParams):

        self.dataset = dataset

        self.imageDataGeneratorParams = imageDataGeneratorParams
        self.dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(**imageDataGeneratorParams)

        self.dataGenerator.fit(np.array(self.dataset[0]))

    def getFlwr(self, batchSize = 32, shuffle = True, sampleWeight = None, seed = 42, saveToDir = None, savePrefix = None, saveFormat = None, subset = None):

        X_train = np.array(self.dataset[0])
        y_train = self.dataset[1]

        it = self.dataGenerator.flow(X_train, y_train, batchSize, shuffle, sampleWeight, seed, saveToDir, savePrefix, saveFormat, subset)

        return it

