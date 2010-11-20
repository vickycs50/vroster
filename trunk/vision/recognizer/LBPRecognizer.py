import numpy
import scipy.spatial.distance as distance

from BaseRecognizer import *

class LBPRecognizer(BaseRecognizer):
	
	def __init__(self, matlab):
		self.matlab = matlab
		self.X = []
		
	def __compute(self, img):
		#img = img*255
		#mapping = self.matlab.getmapping([8.0, 'riu2'], 1)[0]
		#return self.matlab.lbp([img, 1.0, 8.0, mapping, 'h'], 1)[0]
		return self.matlab.lbp(img, 1)[0]
		
	def update(self, image):
		self.X.append(self.__compute(image))
		
	def query(self, image):
		img = self.__compute(image)
		res = []
		for x in self.X:
			res.append(distance.euclidean(x, img))
		
		if len(res)==0:
			res = [0]

		return numpy.average(res)