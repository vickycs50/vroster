import numpy
import cv

from BaseRecognizer import *

## Attempt to use SURF descriptor as a recognizer... did not work
class SURFRecognizer(BaseRecognizer):
	
	def __init__(self, data):
		self.desc = []
		self.points = []
		self.weights = []
		
		self.__update(data, [1]*len(data))
		
	def update(self, data, weights = None):
		if weights == None:
			weights = [1]*len(data)
		self.__update([data], [weights])

		
	def __update(self, data, weights):
		for d in data:
			curr = self.__doSURF(d)
			
			if curr != None and len(curr[0]) != 0:
				self.points.extend(curr[0])
				self.desc.extend(curr[1])
		
		self.weights.extend(weights)
		
	def __doSURF(self, data):
		s = int(math.sqrt(data.size))
		
		d = numpy.reshape(numpy.cast['uint8'](data*255.0), (s,s))
		img = cv.fromarray(d)
		
		try:
			(currPoints, currDesc) = cv.ExtractSURF(img, None, cv.CreateMemStorage(), (1, 400, 3, 4))
			points = []
			for p in currPoints:
				points.append([p[0][0], p[0][1], p[1], p[2], p[3], p[4]])

			return [points, currDesc]
		except Exception as e:
			pass
		return None
				
	def query(self, data):
		dist = []

		d = self.__doSURF(data)
		if d != None:
			for n in range(0, len(self.desc)):
				dist.append(self.__query(n, d)*self.weights[n])
				return numpy.average(dist)
		return 0
		
	def __query(self, i, data):
		dist = numpy.zeros((len(self.desc[i]), len(data)))

		for m in range(0, len(self.desc[i])):
			for n in range(0, len(data)):
				dist[m, n] = distance.euclidean(self.desc[i][m], data[n])	
		
		return numpy.average(numpy.min(dist, axis=1))
