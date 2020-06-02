from KerasImagePreprocessor import KerasImagePreprocessor
from DatasetLoader import DatasetLoader
import numpy as np
import tensorflow as tf

imageDataGeneratorParams = dict( \
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=True,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1.0/255,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None
)

DATASET_PATH = "../datasets/gtsrb/GTSRB/Final_Training/Images/"
LABELS_PATH = "../datasets/gtsrb/labels.txt"


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

datasetLoader = DatasetLoader(DATASET_PATH, imgSize = (64, 64))
dataset1 = datasetLoader.getDataset()
dataset = (dataset1[0][:2], dataset1[1][:2])

#imgPreproc = KerasImagePreprocessor(dataset, **imageDataGeneratorParams)

#imgFlwr = imgPreproc.getFlwr(saveToDir = "./preprocessingNotebookTest", savePrefix = "a", saveFormat = "jpeg")
#imgFlwr.next()