# Class that loads the image data using OpenCV and returns an array used to preprocess it.
import cv2
from pathlib import Path

class DatasetLoader():

	def __init__(self, datasetPath, imgSize = None, interpolation = cv2.INTER_AREA):

		self.X_train = []
		self.y_train = []

		datasetPath = Path(datasetPath)

		for labelDir in datasetPath.iterdir():
			for imgPath in labelDir.iterdir():
				if imgPath.suffix != ".ppm":
					continue

				labelName = int(str(labelDir.name))
				self.y_train.append(labelName)

				img = cv2.imread(str(imgPath), cv2.IMREAD_UNCHANGED)

				self.X_train.append(img)

		if len(self.X_train) == 0 or len(self.y_train) == 0:
			raise Exception("Didn't read any data...")

		if imgSize:
			self.X_train = self.resizeImages(imgSize, interpolation)

		else:
			self.imageSize = self.checkImagesSize()

	def resizeImages(self, imgSizeTuple, interpolation = cv2.INTER_AREA):

		self.imageSize = imgSizeTuple

#		print("Resizing images to: " + str(imgSizeTuple))

		newImages = []

		for img in self.X_train:
			newImages.append(cv2.resize(img, imgSizeTuple, interpolation))

		return newImages


	def checkImagesSize(self):

		defaultSize = self.X_train[0].shape[:2]

		for img in self.X_train:
			if img.shape[:2] != defaultSize:
				return (0, 0)

		return defaultSize


	def getDataset(self):
		return self.X_train, self.y_train

	def getimagesSizes(self):
		return self.imageSize


	def getLabels(self, pathToLabels):

		self.labels = []

		with open(pathToLabels, "r") as fp:
		    lines = fp.readlines()
		    for label in lines:
		        self.labels.append(label.strip())

		return self.labels