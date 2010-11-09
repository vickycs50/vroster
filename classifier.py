import cv
import numpy
import mdp.nodes as mdp
import sys
import scipy.spatial.distance as distance
import math 
import util

class Classifier:
	
	def update(self, data):
		pass
	
	def query(self, data):
		pass
		
class PCA(Classifier):
	
	def __init__(self, data):
		self.X = data
		self.pca = None
		self.__update()
		
	def update(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))

		self.X = numpy.vstack((self.X, data))
		self.__update();
		
	def __update(self):
		self.pca = mdp.PCANode(output_dim=.95)
		self.pca.train(self.X)
		
	def query(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))
	
		return self.pca(data)
	
class Dist(Classifier):
	
	def __init__(self, data):
		self.X = data
		self.__update()
		
	def update(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))

		self.X = numpy.vstack((self.X, data))
		#self.__update();

	def __update(self):
		self.m = numpy.average(self.X)
	
	def query(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))
		return distance.euclidean(self.m, data)
		
class SURF(Classifier):
	
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

class Template(Classifier):
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
