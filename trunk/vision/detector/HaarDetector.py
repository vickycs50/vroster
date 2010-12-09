import cv

from BaseDetector import *


class HaarDetector(BaseDetector):
	"""OpenCV Haar detector implementation"""
	
	def __init__(self, cascade, size):
		self.cascade = cv.Load(cascade)
		self.size = size
		
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
			
		return res
		
class FastHaarDetector(HaarDetector):
	
	def __init__(self, cascade, size, mod=[2, 2]):
		HaarDetector.__init__(self, cascade, size)
		self.mod = mod
		self.ind = [0, 0]
		
	def detect(self, image):
		size = cv.GetSize(image)
		ss = (size[0]/self.mod[0], size[1]/self.mod[1])
		currSlice = (ss[0]*self.ind[0], ss[1]*self.ind[1], ss[0], ss[1])

		self.ind[0] += 1
		if self.ind[0]==self.mod[0]:
			self.ind[0] = 0
			self.ind[1] +=1
		if self.ind[1]==self.mod[1]:
			self.ind[0] = 0
			self.ind[1] = 0
			
		img = cv.GetSubRect(image, currSlice)
		res = HaarDetector.detect(self, img)

		for i in range(0, len(res)):
			res[i] = [res[i][0]+currSlice[0], res[i][1]+currSlice[1], res[i][2], res[i][3]]
			
		return res