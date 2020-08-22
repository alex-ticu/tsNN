import math
import numpy as np
import cv2

class edge():
	def __init__(self, pixela, pixelb, pixelaVal, pixelbVal):
		self.pixela = pixela
		self.pixelb = pixelb
		self.pixelaVal = pixelaVal
		self.pixelbVal = pixelbVal
		self.w = self.calculateEdgeWeight()

	def calculateEdgeWeight(self):

		d1 = self.pixelaVal[0] - self.pixelbVal[0]
		d2 = self.pixelaVal[1] - self.pixelbVal[1]
		d3 = self.pixelaVal[2] - self.pixelbVal[2]

		return math.sqrt(d1**2 + d2**2 + d3**2)


class uni_elts():
	def __init__(self, rank, p, size):
		# Rank of the node
		self.rank = rank
		# Parent of the node
		self.p = p
		# size of the component.
		self.size = size


# Disjoint-set forest w/ union-by-rank and path compression used for generating the graphs / components
class disjoinForestSet():

	# When initialized, every pixel is in it's own component.
	def __init__(self, elements):
		self.num = elements
		self.elts = []

		for i in range(elements):
			elt = uni_elts(0, i, 1)
			self.elts.append(elt)

	# Finding the representative component (the root of the tree).
	def find(self, x):
		y = x
		while y != self.elts[y].p:
			y = self.elts[y].p
		# Path compression
		self.elts[x].p = y
		return y

	# Join two components together based on rank. If the rank is equal, increment one element's rank.
	def join(self, x, y):
		if self.elts[x].rank > self.elts[y].rank:
			self.elts[y].p = x
			self.elts[x].size += self.elts[y].size
		elif self.elts[y].rank > self.elts[x].rank:
			self.elts[x].p = y
			self.elts[y].size += self.elts[x].size
		elif self.elts[x].rank == self.elts[y].rank:
			self.elts[y].rank += 1
			self.elts[x].p = y
			self.elts[y].size += self.elts[x].size

		self.num = self.num - 1

	def num_sets(self):
		return self.num

	def size(self, x):
		return self.elts[x].size


def getWeight(edge):
	return edge.w

def thresholdFunc(size, c):
	return c/size

def segmentGraph(numVertecies, numEdges, edges, c):

	edges.sort(key=getWeight)

	u = disjoinForestSet(numVertecies)
	threshold = []

	# Init thresholds
#	print("Init treshold: " + str(thresholdFunc(1, c)))

	for i in range(numVertecies):
		threshold.append(thresholdFunc(1, c))

	for i in range(numEdges):
		pedge = edges[i]

		a = u.find(pedge.pixela)
		b = u.find(pedge.pixelb)

		if a != b:
			if (pedge.w <= threshold[a]) and (pedge.w <= threshold[b]):
				u.join(a, b)
				a = u.find(a)
				threshold[a] = pedge.w + thresholdFunc(u.size(a), c)

	print("Num of initial regions: " + str(u.num_sets()))
	return u


def segmentImage(image, sigma, c, minSize):
	
	img = cv2.imread(image, cv2.IMREAD_COLOR)

	# Change type from uint8 to int16
	img = np.int16(img)

	height, width = img.shape[:2]
	print("height x width:" + str(height*width))

	img = cv2.GaussianBlur(img, (5,5), sigma)

	edges = []
	for y in range(height):
		for x in range(width):
			if x < width - 1:
				edgea = x + y * width
				edgeb = (x + 1) + y * width
				newEdge = edge(edgea, edgeb, img[y][x], img[y][x+1])
				edges.append(newEdge)

			if y < height - 1:
				edgea = x + y * width
				edgeb = x + (y + 1) * width
				newEdge = edge(edgea, edgeb, img[y][x], img[y+1][x])
				edges.append(newEdge)

			if x < width - 1 and y < height - 1:
				edgea = x + y * width
				edgeb = (x + 1) + (y + 1) * width
				newEdge = edge(edgea, edgeb, img[y][x], img[y+1][x+1])
				edges.append(newEdge)

			if x < width - 1 and y > 0:
				edgea = x + y * width
				edgeb = x + 1 + (y - 1) * width
				newEdge = edge(edgea, edgeb, img[y][x], img[y-1][x+1])
				edges.append(newEdge)

	num = len(edges)
	print("Created " + str(num) + " edges.")

	u = segmentGraph(width*height, num, edges, c)
	
	# Post process small components
	for i in range(num):
		# Find in which component the first pixel is in
		a = u.find(edges[i].pixela)
		# Find in which component the second pixel is in
		b = u.find(edges[i].pixelb)

		# If they are in different components...
		if (a != b) and ((u.size(a) < minSize) or (u.size(b) < minSize)):
			u.join(a, b)

	neighbours = []
	for i in range(num):

		a = u.find(edges[i].pixela)
		b = u.find(edges[i].pixelb)
		if a != b:
			neighbour = (a, b)
			if neighbour not in neighbours:
				neighbours.append(neighbour)

	print(str(neighbours))

	numCCS = u.num_sets()
	print("Created " + str(numCCS) + " components.")

	output = np.zeros((height, width, 3), np.uint8)

	colors = []
	np.random.seed(42)
	for i in range(width * height):
		r = np.random.randint(0, 256)
		g = np.random.randint(0, 256)
		b = np.random.randint(0, 256)
		colors.append(np.array([b, g, r]))

	comps = []
	for y in range(height):
		for x in range(width):
			comp = u.find(x + y * width)
			output[y][x] = colors[comp]
			if comp not in comps:
				comps.append(comp)
#				print("Component: " + str(comp))

	print("Components: " + str(len(comps)))
	for comp in comps:
		for neighbour in neighbours:
			if comp in neighbour:
				print("Component: " + str(neighbour[0]) + " " + str(colors[neighbour[0]]) + " and component:  " + str(neighbour[1]) + " " + str(colors[neighbour[1]]) + " are neighbours!" )



	return output

if __name__ == "__main__":
	image = segmentImage("testImg.ppm", 1, 5000, 2000)
	cv2.imwrite("output2.ppm", image)
