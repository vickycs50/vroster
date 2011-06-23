import os
import cv
import sys
import numpy

from ..util import Image
from BaseRecognizer import *

## Bag recognizer collects all face recognizers into one. Handingling opening of face database folder and managing the recognition results.
class BagRecognizer(BaseRecognizer):
	
	def __init__(self, directory, classifiers, size):
		self.classifiers = classifiers
		self.dist = None
		
		if getattr(directory, '__iter__', False) == False:
			directory = (directory, )

		cid = 0
		
		for currentDir in directory:
			directories = os.listdir(currentDir)
			list.sort(directories)
			
			for d in directories:
				if d[0]!='.':
				
					#print 'Loading recognizer [%02d]: %s\t%s'%(cid, currentDir, d)
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
					

					cid += 1
		for i in range(cid, len(self.classifiers)):
			self.classifiers.pop()

	def query(self, image):
		res = []
		
		for c in self.classifiers:
			res.append(c.query(image))

		return res
		
	def distance(self):
		if self.dist == None:
			self.dist = numpy.zeros((len(self.classifiers), len(self.classifiers)))
			for i in range(0, len(self.classifiers)):
				for j in range(0, len(self.classifiers)):
					a = numpy.average(self.classifiers[i].X, axis=0)
					b = numpy.average(self.classifiers[j].X, axis=0)
					self.dist[i,j] = numpy.linalg.norm(a-b)
		return self.dist
		
	