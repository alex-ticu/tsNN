from felzenszwalb import segmentGraph
from felzenszwalb import edge
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import time

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
			region = Region(uRegions[i], regionPixelIdxs, regionPixelVals, True)
			region.textureHist = region.getTextureHist(self.image)
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

		for neighbours in self.neighboursList:

			neighbours.colorSimilarity = neighbours.getColorSimilarity()

			neighbours.similarity = neighbours.similarity + neighbours.colorSimilarity

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
			print("Index max sim: " + str(S.index(highestSimilarity)))
			print(type(neighbours))
			regionA = neighbours.regionA
			regionB = neighbours.regionB

			self.regions.remove(regionA)
			self.regions.remove(regionB)

			self.u.join(regionA.regionID, regionB.regionID)

			newRegion = Region(self.u.find(regionA.regionID), regionA.pixelIdxs + regionB.pixelIdxs, regionA.pixelVals + regionB.pixelVals)
			newRegion.colorHistogram = (self.u.size(regionA.regionID) * regionA.colorHistogram + self.u.size(regionB.regionID) * regionB.colorHistogram) / (self.u.size(regionA.regionID) + self.u.size(regionB.regionID))

			self.regions.append(newRegion)

			newNeighboursList = []
#			i = 0
			print("Initial process: " + str(time.time() - timer))
			time1 = time.time()
			for nghbrs in self.neighboursList:
#				print("!")
				someNeighbours = (nghbrs.regionA, nghbrs.regionB)
#				print("1!")
				if regionA in someNeighbours and regionB in someNeighbours:
					S.remove(nghbrs.similarity)
					continue
#				print("!!")
				if regionA not in someNeighbours and regionB not in someNeighbours:
					newNeighboursList.append(nghbrs)
#					i = i + 1
#					print("2!!")
#					print("First if: " + str(time.time() - time1))
				else:
#					print("!!!")
					if regionA == someNeighbours[0]:
						time2 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[1])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.similarity = neighbour.colorSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)
#						S.insert(i, neighbour.colorSimilarity)
#						i = i + 1
#						print("3!!!" + str(time.time() - time2))
					elif regionA == someNeighbours[1]:
						time3 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[0])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.similarity = neighbour.colorSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)
#						S.insert(i, neighbour.colorSimilarity)
#						i = i + 1
#						print("4!!!! " + str(time.time() - time3))
					if regionB == someNeighbours[0]:
						time4 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[1])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.similarity = neighbour.colorSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)
#						S.insert(i, neighbour.colorSimilarity)
#						i = i + 1
#						print("5!!!!!" + str(time.time() - time4))
					elif regionB == someNeighbours[1]:
						time5 = time.time()
						S.remove(nghbrs.similarity)
						neighbour = Neighbours(newRegion, someNeighbours[0])
						neighbour.colorSimilarity = sum([min(a, b) for a, b in zip(neighbour.regionA.colorHistogram, neighbour.regionB.colorHistogram)])
						neighbour.similarity = neighbour.colorSimilarity
						newNeighboursList.append(neighbour)
						S.append(neighbour.similarity)
#						S.insert(i, neighbour.colorSimilarity)
#						i = i + 1
#						print("6!!!!!!" + str(time.time() - time5))
#			print("i: " + str(i))

			self.neighboursList = newNeighboursList

			print("Whole process took: " + str(time.time() - timer))

#			output = self.paintRegions(BB = True)
#			saveStr = "output" + str(itera) + ".ppm"
#			cv2.imwrite(saveStr, output)

			itera = itera + 1




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


	def getColorSimilarity(self):

		aHist = self.regionA.colorHistogram
		bHist = self.regionB.colorHistogram

		s = sum([min(a, b) for a, b in zip(aHist, bHist)])

		return s


class Region():

	def __init__(self, regionID, pixelIdxs, pixelVals, initColorHist = False):
		
		self.regionID = regionID
		self.pixelIdxs = pixelIdxs
		self.pixelVals = pixelVals
		self.boundingBox = self.getBoundingBox()
		self.textureHist = 0

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


if __name__ == "__main__":

	image = cv2.imread("testImg.ppm", cv2.IMREAD_UNCHANGED)
	image = np.int16(image)

	ss = SelectiveSearch(image, 1, 5000, 2000)

	output = ss.paintRegions(BB = True)
	cv2.imwrite("output.ppm", output)

	ss.HierarhicalGrouping()
