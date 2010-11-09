import cv
import util
import numpy
import scipy.spatial.distance as distance
import sys

class Detector:
	
	def detect(self, image):
		pass

class Tracker:

	def __init__(self):
		self.objects = None


	def track(self, detected):
		if self.objects == None and len(detected)>0:
			self.objects = numpy.matrix(detected)
		if self.objects == None:
			return []
			
		for d in detected:
			center = self.__center(d)
			
			distances = []
			for n in range(0, self.objects.shape[0]):
				currentCenter = self.__center(numpy.asarray(self.objects[n,:])[0])
				distances.append(distance.euclidean(currentCenter, center))
			
			if numpy.min(distances)<50:
				self.objects[numpy.argmin(distances)] = d
			else:
				self.objects = numpy.vstack((self.objects, d))
				
		return self.__results()
		
	def __results(self):
		res = []
		for n in range(0, self.objects.shape[0]):
			a = numpy.asarray(self.objects[n,:])
			if len(a[0])==2:
				res.append(tuple(a[0]))
		return res
		
		
	def __center(self, d):
		return (d[0]+d[2]/2, d[1]+d[3]/2)
	
class HaarDetector(Detector):
	
	def __init__(self, cascade, size):
		self.cascade = cv.Load(cascade)
		self.size = size
		self.tracker = Tracker()
		
	def detect(self, image):
		haarResults = cv.HaarDetectObjects(image, self.cascade, cv.CreateMemStorage(0), 1.2, 1, cv.CV_HAAR_DO_CANNY_PRUNING, self.size)
		res = []
		for r in haarResults:
			a = list(r[0])
			a[0] -= 10
			a[1] -= 10
			a[2] += 10
			a[3] += 10
			res.append(a)
			
		return self.tracker.track(res)
	
		
		
		
		
		
		
