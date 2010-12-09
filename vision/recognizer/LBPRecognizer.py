import numpy
import scipy.spatial.distance as distance
from math import *
import mdp

from BaseRecognizer import *

class LBPRecognizer(BaseRecognizer):
	
	def __init__(self, matlab=None):
		self.matlab = matlab
		self.X = []
		
	def __compute(self, img):
		cellSize = [float(12), float(12)]
		radius = 1
		
		size = img.shape 
		cells = [ceil(size[0]/float(cellSize[0])), ceil(size[1]/float(cellSize[1]))]
		res = numpy.zeros((cells[0], cells[1], 2**(radius*2+1)**2))

		
		for y in range(size[0]):
			for x in range(size[1]):
				cx = floor(x/cellSize[0])
				cy = floor(y/cellSize[1])
				
				num = 0
				val = img[y, x]
				count = 0
				
				for ty in range(y-radius, y+radius+1):		
					for tx in range(x-radius, x+radius+1):
						if tx>=0 and tx<size[1] and ty>=0 and ty<size[0]:
							if val>img[ty, tx]:
								num += 2**count
						count += 1
						
				res[cy, cx, num] += 1 

		return res.ravel()	
		
		#return self.matlab.lbp(img, 1)[0]
		
	def update(self, image):
		self.X.append(self.__compute(image))
		
		#self.pca = mdp.nodes.PCANode(output_dim=.9)
		#x = numpy.cast[float](numpy.matrix(self.X))
		#self.pca.train(x)

		
	def query(self, image):
		img = numpy.cast[float](numpy.matrix(self.__compute(image)))
		
		res = []
		for x in self.X:
			res.append(distance.euclidean(x, img))
		
		if len(res)==0:
			res = [0]
        
		return numpy.average(res)
		
		#y = self.pca(img)
		#return numpy.power(y,2).sum()*-1
