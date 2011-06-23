import cv
import threading
import time
from BaseDetector import *
from HaarDetector import *

## Attempt to run the face detector as a seperate thread to increase efficiency
class ThreadedDetector:	
	def __init__(self, cascade, size):
		self.past = []
		
		self.haar = HaarDetector(cascade, size)
		self.haarResult = None
		self.thread = None
		
	def detect(self, image):
		res = None

		if len(self.past) == 0:
			res = self.haar.detect(image)
		else:
			if self.thread == None or self.thread.is_alive()==False:
				if self.haarResult != None:

					res = self.haarResult
					self.haarResult = None
				self.image = cv.CloneImage(image)
				self.thread = threading.Thread(target=ThreadedDetector.doThreadedHaar, args=(self, image))
				self.thread.daemon = True
				self.thread.start()
			if res == None:
				res = self.past[len(self.past)-1]

		self.past.append(res)
		return res

	def doThreadedHaar(self, image):
		self.haarResult = self.haar.detect(self.image)
		#return self.haarResult