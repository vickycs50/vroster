import cv

from BaseVideo import *

class CVBaseVideo(BaseVideo):
	
	def __init__(self, source):
		self.source = source
	
	def next(self):
		return cv.QueryFrame(self.source)
		
	def skip(self, index):
		pass
		
	def length(self):
		pass
		
class CVFileVideo(CVBaseVideo):

	def __init__(self, filename):
		CVBaseVideo.__init__(self, cv.CaptureFromFile(filename))
		
class CVCamVideo(CVBaseVideo):
	
	def __init__(self):
		CVBaseVideo.__init__(self, cv.CaptureFromCAM(0))