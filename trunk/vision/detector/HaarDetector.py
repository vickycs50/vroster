import cv

from BaseDetector import *


class HaarDetector(BaseDetector):
	
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