import os
import cv
import numpy

from ..util import Image
from BaseRecognizer import *

class BagRecognizer(BaseRecognizer):
	
	def __init__(self, directory, classifiers, size):
		self.classifiers = classifiers
		
		directories = os.listdir(directory)
		
		cid = 0
		
		for d in directories:
			if d[0]!='.':
				
				print 'Loading:',d 
				files = os.listdir(directory+'/'+d)
				
				for file in files:
					if file[0] != '.':
						image = cv.LoadImage(directory+'/'+d+'/'+file, cv.CV_LOAD_IMAGE_GRAYSCALE)
						image2 = cv.CreateImage(size, image.depth, 1)
						cv.Resize(image, image2)
						x = Image.cv2array(image2)/255.0
						
						self.classifiers[cid].update(x)
			
				cid += 1

	def query(self, image):
		res = []
		
		for c in self.classifiers:
			res.append(c.query(image))

		return res