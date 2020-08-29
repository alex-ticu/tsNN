from felzenszwalb import segmentGraph
from felzenszwalb import edge
import cv2
import numpy as np
#from skimage.feature import local_binary_pattern
import time
import random

class SelectiveSearch():

	def __init__(self, image, sigma, c, minSize = 0):

		self.sigma = sigma
		self.c = c
		self.minSize = minSize
		print("Applying gaussian blur to image...")
		self.image = self.applyGaussianBlur(image)
		print("Done!")

		self.height = image.shape[0]
		self.width = image.shape[1]
		self.imageChannels = image.shape[2]

		print("Calculating edge weights...")
		self.edges = self.createEdges()
		print("Created: " + str(len(self.edges)) + " edges.(" + str(self.numEdges) + ")")

		print("Segmenting graph...")
		self.u = segmentGraph(self.height*self.width, self.numEdges, self.edges, self.c)

		# Postprocess regions to be at least minSize pixels in size.
		print("Postprocessing graph...")
		for i in range(self.numEdges):
			a = self.u.find(self.edges[i].pixela)
			b = self.u.find(self.edges[i].pixelb)

			if (a != b) and ((self.u.size(a) < minSize) or (self.u.size(b) < minSize)):
				self.u.join(a, b)
		print("Regions after postprocessing: " + str(self.u.num_sets()))

		self.regions, self.neighboursList = self.getRegions()


	def applyGaussianBlur(self, image, filterSize=(5,5)):
		return cv2.GaussianBlur(image, filterSize, self.sigma)


	def createEdges(self):

		edges = []
		for y in range(self.height):
			for x in range(self.width):

				if x < self.width - 1:
					edgea = x + y * self.width
					edgeb = (x + 1) + y * self.width
					newEdge = edge(edgea, edgeb, self.image[y][x], self.image[y][x+1])
					edges.append(newEdge)

				if y < self.height - 1:
					edgea = x + y * self.width
					edgeb = x + (y + 1) * self.width
					newEdge = edge(edgea, edgeb, self.image[y][x], self.image[y+1][x])
					edges.append(newEdge)

				if x < self.width - 1 and y < self.height - 1:
					edgea = x + y * self.width
					edgeb = (x + 1) + (y + 1) * self.width
					newEdge = edge(edgea, edgeb, self.image[y][x], self.image[y+1][x+1])
					edges.append(newEdge)

				if x < self.width - 1 and y > 0:
					edgea = x + y * self.width
					edgeb = x + 1 + (y - 1) * self.width
					newEdge = edge(edgea, edgeb, self.image[y][x], self.image[y-1][x+1])
					edges.append(newEdge)

		self.numEdges = len(edges)

		return edges

	def getRegions(self):
		
		uRegions = []
		pixelIdxs = [[] for i in range(self.u.num_sets())]
		pixelVals = [[] for i in range(self.u.num_sets())]
		neighboursList = []

		for y in range(self.height):
			for x in range(self.width):

				pixelPos = x + y * self.width
				a = self.u.find(pixelPos)

				if a not in uRegions:
					uRegions.append(a)
				pixelIdxs[uRegions.index(a)].append([x, y])
				pixelVals[uRegions.index(a)].append(self.image[y][x])

				if x < self.width - 1:
					
					b = self.u.find((x + 1) + y*self.width)
					if a != b:
						neighbours = (a, b)
						if neighbours not in neighboursList:
							neighboursList.append(neighbours)

				if y < self.height - 1:

					b = self.u.find(x + (y + 1) * self.width)
					if a != b:
						neighbours = (a, b)
						if neighbours not in neighboursList:
							neighboursList.append(neighbours)

				if x < self.width - 1 and y < self.height - 1:

					b = self.u.find((x + 1) + (y + 1) * self.width)
					if a != b:
						neighbours = (a, b)
						if neighbours not in neighboursList:
							neighboursList.append(neighbours)

				if x < self.width - 1 and y > 0:

					b = self.u.find(x + 1 + (y - 1) * self.width)
					if a != b:
						neighbours = (a, b)
						if neighbours not in neighboursList:
							neighboursList.append(neighbours)

		regions = []
		for i in range(len(uRegions)):
			regionPixelIdxs = [pixelIdx for pixelIdx in pixelIdxs[i]]
			regionPixelVals = [pixelVal for pixelVal in pixelVals[i]]
			region = Region(uRegions[i], regionPixelIdxs, regionPixelVals, self.image, True, True)
#			region.textureHistogram = region.getTextureHist(self.image)
			regions.append(region)

		newNeighboursList = []
		for i in range(len(neighboursList)):
			a = neighboursList[i][0]
			b = neighboursList[i][1]

			for region in regions:
				if a == region.regionID:
					regionA = region
				if b == region.regionID:
					regionB = region
			newNeighboursList.append(Neighbours(regionA, regionB))

		neighboursList = newNeighboursList

		print("Created " + str(len(regions)) + " Region classes.")
		print("Created a list of " + str(len(neighboursList)) + " neighbours.")
		return regions, neighboursList


	def HierarhicalGrouping(self):

		S = []
		boxes = []

		for region in self.regions:
			boxes.append(region.boundingBox)

		for neighbours in self.neighboursList:

			neighbours.colorSimilarity = neighbours.getColorSimilarity()
			neighbours.textureSimilarity = neighbours.getTextureSimilarity()
			neighbours.sizeSimilarity = neighbours.getSizeSimilarity(self.width * self.height, self.u)
			neighbours.fillSimilarity = neighbours.getFillSimilarity(self.width * self.height, self.u)

			neighbours.similarity = neighbours.similarity + neighbours.colorSimilarity + neighbours.textureSimilarity + neighbours.fillSimilarity

			S.append(neighbours.similarity)



		itera = 0
		while S:

			timer = time.time()
			print("Iter: " + str(itera))
			print("Len S: " + str(len(S)))
			print("Len neighbours: " + str(len(self.neighboursList)))
			print("Regions: " + str(len(self.regions)))
			highestSimilarity = max(S)
			print("Max sim: " + str(highestSimilarity))

			for nghbrs in self.neighboursList:
				if nghbrs.similarity == highestSimilarity:
					neighbours = nghbrs
					break

			regionA = neighbours.regionA
			regionB = neighbours.regionB

			self.regions.remove(regionA)
			self.regions.remove(regionB)

			self.u.join(regionA.regionID, regionB.regionID)

			newRegion = Region(self.u.find(regionA.regionID), regionA.pixelIdxs + regionB.pixelIdxs, regionA.pixelVals + regionB.pixelVals)
			newRegion.colorHistogram = (self.u.size(regionA.regionID) * regionA.colorHistogram + self.u.size(regionB.regionID) * regionB.colorHistogram) / (self.u.size(regionA.regionID) + self.u.size(regionB.regionID))
			newRegion.textureHistogram = (self.u.size(regionA.regionID) * regionA.textureHistogram + self.u.size(regionB.regionID) * regionB.textureHistogram) / (self.u.size(regionA.regionID) + self.u.size(regionB.regionID))
			self.regions.append(newRegion)

			boxes.append(newRegion.boundingBox)

			newNeighboursList = []

			print("Initial process: " + str(time.time() - timer))
			time1 = time.time()
			for nghbrs in self.neighboursList:

				someNeighbours = (nghbrs.regionA, nghbrs.regionB)

				if regionA in someNeighbours and regionB in someNeighbours:
					S.remove(nghbrs.similarity)
					continue

				if regionA not in someNeighbours and regionB not in someNeighbours:
					newNeighboursList.append(nghbrs)

				else:
					if regionA == someNeighbours[0]:
						time2 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[1])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.textureSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.textureHistogram, neighbour.regionB.textureHistogram)])
						neighbour.sizeSimilarity = neighbour.getSizeSimilarity(self.width * self.height, self.u)
						neighbour.fillSimilarity = neighbour.getFillSimilarity(self.width * self.height, self.u)
						neighbour.similarity = neighbour.colorSimilarity + neighbour.textureSimilarity + neighbour.sizeSimilarity + neighbour.fillSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)

					elif regionA == someNeighbours[1]:
						time3 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[0])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.textureSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.textureHistogram, neighbour.regionB.textureHistogram)])
						neighbour.sizeSimilarity = neighbour.getSizeSimilarity(self.width * self.height, self.u)
						neighbour.fillSimilarity = neighbour.getFillSimilarity(self.width * self.height, self.u)
						neighbour.similarity = neighbour.colorSimilarity + neighbour.textureSimilarity + neighbour.sizeSimilarity + neighbour.fillSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)

					if regionB == someNeighbours[0]:
						time4 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[1])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.textureSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.textureHistogram, neighbour.regionB.textureHistogram)])
						neighbour.sizeSimilarity = neighbour.getSizeSimilarity(self.width * self.height, self.u)
						neighbour.fillSimilarity = neighbour.getFillSimilarity(self.width * self.height, self.u)
						neighbour.similarity = neighbour.colorSimilarity + neighbour.textureSimilarity + neighbour.sizeSimilarity + neighbour.fillSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)

					elif regionB == someNeighbours[1]:
						time5 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[0])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.textureSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.textureHistogram, neighbour.regionB.textureHistogram)])
						neighbour.sizeSimilarity = neighbour.getSizeSimilarity(self.width * self.height, self.u)
						neighbour.fillSimilarity = neighbour.getFillSimilarity(self.width * self.height, self.u)
						neighbour.similarity = neighbour.colorSimilarity + neighbour.textureSimilarity + neighbour.sizeSimilarity + neighbour.fillSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)


			self.neighboursList = newNeighboursList

			print("Whole process took: " + str(time.time() - timer))

#			if itera % 50 == 0:

#				output = self.paintRegions(BB = True)
#				saveStr = "output" + str(itera) + ".ppm"
#				cv2.imwrite(saveStr, output)

			itera = itera + 1

		random.shuffle(boxes)
		print(len(boxes))

		uniqueBoxes = []
		[uniqueBoxes.append(box) for box in boxes if box not in uniqueBoxes]
		print(len(uniqueBoxes))

		return uniqueBoxes

	def paintRegions(self, BB = False):

		colors = []
		np.random.seed(42)
		for i in range(self.width * self.height):
			r = np.random.randint(0, 256)
			g = np.random.randint(0, 256)
			b = np.random.randint(0, 256)
			colors.append(np.array([b, g, r]))

		output = np.zeros((self.height, self.width, 3), np.uint8)
		

		for region in self.regions:
			for pixel in region.pixelIdxs:
				output[pixel[1]][pixel[0]] = colors[region.regionID]

		for region in self.regions:
			if BB:
				topLeft = region.boundingBox[0]
				bottomRight = region.boundingBox[3]

				output = cv2.rectangle(output, topLeft, bottomRight, (255, 0, 0), 2)

		return output


class Neighbours():

	def __init__(self, regionA, regionB):

		self.regionA = regionA
		self.regionB = regionB
		self.similarity = 0
		self.colorSimilarity = 0
		self.textureSimilarity = 0
		self.sizeSimilarity = 0
		self.fillSimilarity = 0


	def getColorSimilarity(self):

		aHist = self.regionA.colorHistogram
		bHist = self.regionB.colorHistogram

		s = sum([min(a, b) for a, b in zip(aHist, bHist)])

		return s

	def getTextureSimilarity(self):

		aHist = self.regionA.textureHistogram
		bHist = self.regionB.textureHistogram

		s = sum([min(a, b) for a, b in zip(aHist, bHist)])


		return s

	def getSizeSimilarity(self, imgSize, u):

		boundingBoxA = self.regionA.boundingBox
		boundingBoxB = self.regionB.boundingBox

		s = 1.0 - (((boundingBoxA[3][0] - boundingBoxA[0][0]) * (boundingBoxA[3][1] - boundingBoxA[0][1]) + (boundingBoxB[3][0] - boundingBoxB[0][0]) * (boundingBoxB[3][1] - boundingBoxB[0][1])) / imgSize)


		return s

	def getFillSimilarity(self, imgSize, u):

		boundingBoxA = self.regionA.boundingBox
		boundingBoxB = self.regionB.boundingBox

		# Bounding box: (topLeft, topRight, bottomLeft, bottomRight)
		minX = 0
		minY = 0
		maxX = 0
		maxY = 0

		if boundingBoxA[0][0] < boundingBoxB[0][0]:
			minX = boundingBoxA[0][0]
		else:
			minX = boundingBoxB[0][0]

		if boundingBoxA[0][1] < boundingBoxB[0][1]:
			minY = boundingBoxA[0][1]
		else:
			minY = boundingBoxB[0][1]

		if boundingBoxA[3][0] > boundingBoxB[3][0]:
			maxX = boundingBoxA[3][0]
		else:
			maxX = boundingBoxB[3][0]

		if boundingBoxA[3][1] > boundingBoxB[3][1]:
			maxY = boundingBoxA[3][1]
		else:
			maxY = boundingBoxB[3][1]

		boundingBoxSize = (maxX - minX) * (maxY - minY)


		s = 1 - ((boundingBoxSize - (boundingBoxA[3][0] - boundingBoxA[0][0]) * (boundingBoxA[3][1] - boundingBoxB[0][1]) - (boundingBoxB[3][0] - boundingBoxB[0][0]) * (boundingBoxB[3][1] - boundingBoxB[0][1])) / imgSize)


		return s



class Region():

	def __init__(self, regionID, pixelIdxs, pixelVals, image = None, initColorHist = False, initTextureHist = False):
		
		self.regionID = regionID
		self.pixelIdxs = pixelIdxs
		self.pixelVals = pixelVals
		self.boundingBox = self.getBoundingBox()
		
		if initTextureHist:
			self.textureHistogram = self.getTextureHist(image)

		if initColorHist:
			self.colorHistogram = self.getColorHist()

	def getColorHist(self):

		BINS = 25
		hist = np.array([])
		channels = len(self.pixelVals[0])

		pixels = np.array(self.pixelVals)
		for channel in range(channels):
			hist = np.concatenate([hist] + [np.histogram(pixels[:, channel], BINS)[0]])

		hist = hist / np.sum(hist)

		return hist


	def getTextureHist(self, image):

		channels = len(self.pixelVals[0])
		height = image.shape[0]
		width = image.shape[1]
		lbpRegion = []

		for pixel in self.pixelIdxs:
			# topLeft -> topRight -> bottomRight -> bottomLeft
			lbp = [[] for i in range(channels)]
			x = pixel[0]
			y = pixel[1]
			for channel in range(channels):
				 
				 if x > 0 and y > 0:
				 	lbp[channel].append(image[y][x][channel] > image[y-1][x-1][channel])
				 else:
				 	lbp[channel].append(False)

				 if y > 0:
				 	lbp[channel].append(image[y][x][channel] > image[y-1][x][channel])
				 else:
				 	lbp[channel].append(False)

				 if x < width - 1 and y > 0:
				 	lbp[channel].append(image[y][x][channel] > image[y-1][x+1][channel])
				 else:
				 	lbp[channel].append(False)

				 if x < width - 1:
				 	lbp[channel].append(image[y][x][channel] > image[y][x+1][channel])
				 else:
				 	lbp[channel].append(False)

				 if x < width - 1 and y < height - 1:
				 	lbp[channel].append(image[y][y][channel] > image[y+1][x+1][channel])
				 else:
				 	lbp[channel].append(False)

				 if y < height - 1:
				 	lbp[channel].append(image[y][x][channel] > image[y+1][x][channel])
				 else:
				 	lbp[channel].append(False)

				 if x > 0 and y < height - 1:
				 	lbp[channel].append(image[y][x][channel] > image[y+1][x-1][channel])
				 else:
				 	lbp[channel].append(False)

				 if x > 0:
				 	lbp[channel].append(image[y][x][channel] > image[y][x-1][channel])
				 else:
				 	lbp[channel].append(False)
			lbpRegion.append(np.array(lbp))

		lbpRegion = np.array(lbpRegion)

		BINS = 10
		hist = np.array([])
		directions = 8

		for channel in range(channels):
			for direction in range(directions):
				hist = np.concatenate([hist] + [np.histogram(lbpRegion[:, channel, direction], BINS)[0]])

		hist = hist / np.sum(hist)
#		print("Texture hist: " + str(hist.shape) + " " + str(hist))

		return hist


	def getBoundingBox(self):

		minX = self.pixelIdxs[0][0]
		minY = self.pixelIdxs[0][1]
		maxX = self.pixelIdxs[len(self.pixelIdxs)-1][0]
		maxY = self.pixelIdxs[len(self.pixelIdxs)-1][1]

		for pixel in self.pixelIdxs:

			if pixel[0] < minX:
				minX = pixel[0]

			if pixel[0] > maxX:
				maxX = pixel[0]

			if pixel[1] < minY:
				minY = pixel[1]

			if pixel[1] > maxY:
				maxY = pixel[1]

		# Bounding box: (topLeft, topRight, bottomLeft, bottomRight)
		boundingBox = [(minX, minY), (maxX, minY), (minX, maxY), (maxX, maxY)]
		return boundingBox


def filterBoxes(boxes, minSize, minRatio = None, topN = None):

	proposal = []

	for box in boxes:
		# Calculate width and height of the box
		w = box[3][0] - box[0][0]
		h = box[3][1] - box[0][1]

		# Filter for size
		if w < minSize or h < minSize:
			continue

		# Filter for box ratio
		if minRatio:
			if w / h < minRatio or h / w < minRatio:
				continue

		proposal.append(box)

	if topN:
		if topN <= len(proposal):
			return proposal[:topN]
		else:
			return proposal
	else:
		return proposal

if __name__ == "__main__":

	image = cv2.imread("testImg.ppm", cv2.IMREAD_UNCHANGED)
	image = np.int16(image)
	output2 = cv2.imread("testImg.ppm", cv2.IMREAD_UNCHANGED)

	ss = SelectiveSearch(image, 1, 2000, 100)

	output = ss.paintRegions(BB = True)
	cv2.imwrite("output.ppm", output)

	boxes = ss.HierarhicalGrouping()
	topNBoxes = filterBoxes(boxes, minSize = 100, minRatio = 0.3)

	for bb in topNBoxes:
		print("Bounding box: " + str(bb))
		topLeft = bb[0]
		bottomRight = bb[3]

		output2 = cv2.rectangle(output, topLeft, bottomRight, (255, 0, 0), 2)

	cv2.imwrite("outBoxes.ppm", output2)
