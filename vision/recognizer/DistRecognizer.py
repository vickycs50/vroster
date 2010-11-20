import numpy
import scipy.spatial.distance as distance

from BaseRecognizer import *

class Dist(BaseRecognizer):
	
	def __init__(self, data):
		self.X = data
		self.__update()
		
	def update(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))

		self.X = numpy.vstack((self.X, data))

	def __update(self):
		self.m = numpy.average(self.X)
	
	def query(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))
		return distance.euclidean(self.m, data)