from selectiveSearch import SelectiveSearch
from selectiveSearch import filterBoxes
import os
import cv2
import numpy as np
import time

DATASET_PATH="../datasets/gtsdb/FullIJCNN2013"
OUTPUT_DIR="./results"

if not os.path.isdir(DATASET_PATH):
	print(DATASET_PATH + " is not a directory. Try downloading the dataset!")
	exit(1)

def runSelectiveSearch(inputImg, output, sigma, c, minsize = 0):
	start = time.time()
	image = cv2.imread(os.path.join(DATASET_PATH, inputImg), cv2.IMREAD_UNCHANGED)
	image = np.int16(image)
	
	ss = SelectiveSearch(image, sigma, c, minsize)
	boxes = ss.HierarhicalGrouping()
	topNBoxes = filterBoxes(boxes, minSize = 100, minRatio = 0.3, topN = 2000)
	
	outputFile = os.path.join(OUTPUT_DIR, inputImg[:-4] + ".csv")

	with open(outputFile, "w") as file:
		for bb in boxes:
			topLeft = bb[0]
			bottomRight = bb[3]
			file.writelines(str(topLeft)[1:-1].replace(" ", "") + "," + str(bottomRight)[1:-1].replace(" ", "") + "\n")
	
	end = time.time()
	print("One run took: " + str(end - start))
	

for i, image in enumerate(sorted(os.listdir("../datasets/gtsdb/FullIJCNN2013"))):
	if ".ppm" in image:
		print("Running selectiveSearch for " + image)
		runSelectiveSearch(image, OUTPUT_DIR, 1, 2000, 100)
