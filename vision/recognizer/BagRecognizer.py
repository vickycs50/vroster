import os
import cv
import numpy

from ..util import Image
from BaseRecognizer import *

class BagRecognizer(BaseRecognizer):
	
	def __init__(self, directory, classifiers, size):
		self.classifiers = classifiers
		
		if getattr(directory, '__iter__', False) == False:
			directory = (directory, )

		cid = 0
		
		for currentDir in directory:
			directories = os.listdir(currentDir)
			list.sort(directories)
			
			for d in directories:
				if d[0]!='.':
				
					#print 'Loading recognizer:',d 
					files = os.listdir(currentDir+'/'+d)
				
					if cid < len(self.classifiers):
						for f in files:
							if f[0] != '.' and f != 'info.txt':
								image = cv.LoadImage(currentDir+'/'+d+'/'+f, cv.CV_LOAD_IMAGE_GRAYSCALE)
								image2 = cv.CreateImage(size, image.depth, 1)
								cv.Resize(image, image2)
								x = Image.cv2array(image2)/255.0
								self.classifiers[cid].update(x)
							if f == 'info.txt':
								self.classifiers[cid].info =  open(currentDir+'/'+d+'/'+f, 'r').read()
					#else:
					#	print 'Skipping..'
					cid += 1
		for i in range(cid, len(self.classifiers)):
			self.classifiers.pop()

	def query(self, image):
		res = []
		
		for c in self.classifiers:
			res.append(c.query(image))

		return res