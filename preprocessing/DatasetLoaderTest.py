import numpy as np
from DatasetLoader import DatasetLoader
import cv2


DATASET_PATH = "../datasets/gtsrb/GTSRB/Final_Training/Images/"
LABELS_PATH = "../datasets/gtsrb/labels.txt"


# TEST 1

print("TEST 1!")

simpleInitialization = DatasetLoader(DATASET_PATH)

dataset = simpleInitialization.getDataset()

imgs, labels = dataset
imgNames = simpleInitialization.getLabels(LABELS_PATH)

for i in np.random.randint(low = 0, high = len(imgs), size = 10):

	cv2.imshow("test", imgs[i])
	print(imgNames[labels[i]])
	cv2.waitKey(0)

# END TEST !

# TEST 2

print("TEST 2!")

sizeInitialization = DatasetLoader(DATASET_PATH, imgSize = (64, 64))

dataset = sizeInitialization.getDataset()

imgs, labels = dataset

imgNames = simpleInitialization.getLabels(LABELS_PATH)

for i in np.random.randint(low = 0, high = len(imgs), size = 10):

	cv2.imshow("test", imgs[i])
	print("Size of image: " + str(imgs[i].shape[:2]))
	print(imgNames[labels[i]])
	cv2.waitKey(0)

# END TEST 2