import numpy

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
			self.objects = numpy.vstack((self.objects, d))
		
		while True:
			obj = []
			
			for i in range(0, self.objects.shape[0]):
				for j in range(0, self.objects.shape[0]):
					if i!=j and i not in obj and j not in obj:
						ic = self.__center(numpy.asarray(self.objects[i,:])[0])
						jc = self.__center(numpy.asarray(self.objects[j,:])[0])
						if numpy.linalg.norm(ic-jc)<50:
							obj.append(j)
							
			if obj == []:
				break
			else:
				new = []
				for i in range(0, self.objects.shape[0]):
					if i not in obj:
						new.append(self.objects[i,:])
				self.objects = numpy.vstack(new)
				
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
		