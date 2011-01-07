import numpy
import scipy.spatial.distance as distance

from BaseTracker import *

class TrivialTracker(BaseTracker):
	
	def __init__(self):
		self.objects = None

	def reset(self):
		self.objects = None

	def update(self, observations):
		
		if self.objects == None and len(observations)>0:
			self.objects = numpy.matrix(observations)
		if self.objects == None:
			return []
		
		for d in observations:
			center = self.__center(d)
			
			distances = []
			for n in range(0, self.objects.shape[0]):
				currentCenter = self.__center(numpy.asarray(self.objects[n,:])[0])
				distances.append(distance.euclidean(currentCenter, center))
			
			if numpy.min(distances)<50:
				self.objects[numpy.argmin(distances)] = d
			else:
				self.objects = numpy.vstack((self.objects, d))
				
		return self.getObjects()
		
	def getObjects(self):
		if self.objects==None:
			return []
			
		res = []
		for n in range(0, self.objects.shape[0]):
			a = numpy.asarray(self.objects[n,:])
			res.append(tuple(a[0]))
		return res
		
		
	def __center(self, d):
		return (d[0]+d[2]/2, d[1]+d[3]/2)
		