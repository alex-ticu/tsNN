import cv2
import os
import sys
import numpy as np
import random

DATASET_PATH="../datasets/gtsdb/FullIJCNN2013"
OUTPUT_DIR="./results"
IMAGE_EXTENSION=".ppm"
CSV_EXTENSION=".csv"
GT_FILE="gt.txt"

if not os.path.isdir(DATASET_PATH):
	print(DATASET_PATH + " is not a directory. Try downloading the dataset!")
	exit(1)

def getGroundTruths():

	truths = []
	for i in range(0, 900):
		truths.append([])

	gtPath = os.path.join(DATASET_PATH, GT_FILE)
	with open(gtPath, "r") as gtFile:
		for line in gtFile.readlines():
			imageNumber = int(line.split(".")[0])
			xTopLeft = int(line.split(";")[1])
			yTopLeft = int(line.split(";")[2])
			xBottomRight = int(line.split(";")[3])
			yBottomRight = int(line.split(";")[4])
			classID = int(line.split(";")[5].strip())
			topLeft = (xTopLeft, yTopLeft)
			bottomRight = (xBottomRight, yBottomRight)
			truths[imageNumber].append((topLeft, bottomRight, classID))

	return truths


def getIOU(box1, box2):

	x1 = max(box1[0][0], box2[0][0])
	y1 = max(box1[0][1], box2[0][1])
	x2 = min(box1[1][0], box2[1][0])
	y2 = min(box1[1][1], box2[1][1])

	intersectionArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
	if (intersectionArea == 0):
		return 0

	box1Area = (box1[0][0] - box1[1][0] + 1) * (box1[0][1] - box1[1][1] + 1)
	box2Area = (box2[0][0] - box2[1][0] + 1) * (box2[0][1] - box2[1][1] + 1)

	iou = intersectionArea / float(box1Area + box2Area - intersectionArea)

	return iou


def parseResults(resultsPath, datasetPath, imageToCheck, groundTruths):

	imagePath = os.path.join(datasetPath, imageToCheck + IMAGE_EXTENSION)
	print(imagePath)
	image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
	if image is None:
		sys.exit("Could not read the image.")
	cv2.imshow("Display window", image)
	cv2.waitKey(0)

	boxes = []
	with open(os.path.join(OUTPUT_DIR, imageToCheck + CSV_EXTENSION), "r") as file:
		for line in file.readlines():
			coords = line.split(",")
			topLeft = (int(coords[0]), int(coords[1]))
			bottomRight = (int(coords[2]), int(coords[3].strip()))
			boxes.append((topLeft, bottomRight))

	output = image.copy()

	for gt in groundTruths:
		output = cv2.rectangle(output, gt[0], gt[1], (50,205,50), 3)
		for box in boxes:
			iou = getIOU(box, gt)
			if iou > 0.5:
				output = cv2.rectangle(output, box[0], box[1], (255,0,0), 3)
			elif iou > 0.1 and iou < 0.3:
				output = cv2.rectangle(output, box[0], box[1], (255, 255, 0), 2)
			output = cv2.rectangle(output, box[0], box[1], (0,0,255), 1)


	cv2.imshow("Display window", output)
	cv2.waitKey(0)


if __name__ == "__main__":

	randomImages = []
	for i in range(0, 10):
		randNumber = str(random.randint(0, 899))
		numString = "0" * (5 - len(randNumber)) + randNumber
		randomImages.append(numString)

	gt = getGroundTruths()
	print(str(gt[0]))
	parseResults(OUTPUT_DIR, DATASET_PATH, "00000", gt[0])

	for img in randomImages:
		print(img)
		parseResults(OUTPUT_DIR, DATASET_PATH, img, gt[int(img)])