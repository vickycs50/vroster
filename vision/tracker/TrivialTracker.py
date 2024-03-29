import numpy

from BaseTracker import *

## Keeps track of objects between frames because the Haar face detector is not stable enough
class TrivialTracker(BaseTracker):
	
	def __init__(self, config):
		self.objects = None
		self.config = config

	def reset(self):
		self.objects = None

	def update(self, observations):
		if self.config.TrackerEnabled == False:
			self.objects = None
		
		if self.objects == None and len(observations)>0:
			self.objects = numpy.matrix(observations)
		if self.objects == None:
			return []
			
		for observation in observations:
			bestDistance = []
			a = self.__center(numpy.asarray(observation))
			
			for i in range(0, self.objects.shape[0]):
				b = self.__center(numpy.asarray(self.objects[i,:])[0])
				bestDistance.append(numpy.linalg.norm(a-b))
			
			x = numpy.argmin(bestDistance)
			if bestDistance[x]<self.config.TrackerDistance:
				self.objects[x, :] += numpy.matrix(observation)
				self.objects[x, :] *= .5
			else:
				self.objects = numpy.vstack((self.objects, observation))
		
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
		return numpy.matrix([d[0]+d[2]/2, d[1]+d[3]/2])
		