import numpy
import cv

from BaseRecognizer import *

## Attempt to use a template based face recognizer... did not work
class TemplateRecognizer(BaseRecognizer):
	def __init__(self, data):
		self.desc = []
		self.weights = []
		self.__update(data, [1]*len(data))
		
		
	def update(self, data, weights=None):
		if weights == None:
			weights = [1]*len(data)
		
		self.__update([data],[weights])

		
	def __update(self, data, weights):
		for d in data:
			img = self.__fixSize(d)
			self.desc.append(img)	
		self.weights.extend(weights)
			
	def query(self, data):
		dist = []

		img = self.__fixSize(data)

		for i in range(0, len(self.desc)):
			desc = self.desc[i]
			res = cv.CreateImage((1,1), cv.IPL_DEPTH_32F, 1)
			cv.MatchTemplate(img, desc, res, cv.CV_TM_CCORR_NORMED)
			res = util.cv2array(res)+1
			dist.append(self.weights[i] * numpy.average(numpy.average(res)))
		return numpy.average(dist)
	
	def __fixSize(self,data):
		if data.ndim==1:
			s = int(math.sqrt(data.size))
			dimg = numpy.reshape(numpy.cast['uint8'](data*255.0), (s,s))
			return cv.fromarray(dimg)
		return cv.fromarray(numpy.cast['uint8'](data*255.0))
