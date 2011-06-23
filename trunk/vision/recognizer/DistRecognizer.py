import numpy
import scipy.spatial.distance as distance

from BaseRecognizer import *

## Basic face recognizer that compares image distances
class Dist(BaseRecognizer):
	
	def __init__(self, data=None):
		self.X = data
		
	def update(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))

		if self.X!=None:
			self.X = numpy.vstack((self.X, data))
		else:
			self.X = data
		self.__update()

	def __update(self):
		self.m = numpy.average(self.X)
	
	def query(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))
		return distance.euclidean(self.m, data)