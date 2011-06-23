import numpy
import scipy.spatial.distance as distance
import scipy.weave as weave
from math import *
import mdp

from BaseRecognizer import *

## LBP based recognizer (worked best)
class LBPRecognizer(BaseRecognizer):
	cache = dict()
	
	def __init__(self, cversion=False):
		BaseRecognizer.__init__(self)
		self.X = []
		self.cversion = cversion
	
	def compute(self, img):
		return self.__compute(img)
		
	def __compute(self, img):
		cellSize = [float(12), float(12)]
		radius = 7
		points = 8
		
		size = img.shape 
		cells = [ceil(size[0]/float(cellSize[0])), ceil(size[1]/float(cellSize[1]))]
		res = numpy.zeros((cells[0], cells[1], 2**points))

		xpoints = radius*numpy.cos(numpy.pi/(points/2) * numpy.array(range(0,points/2)))
		ypoints = radius*numpy.sin(numpy.pi/(points/2) * numpy.array(range(0,points/2)))	
		xpoints = numpy.hstack((xpoints, xpoints))
		ypoints = numpy.hstack((ypoints, -ypoints))
		
		if self.cversion == True:
			code = """
			double sizeY = size[0];
			double sizeX = size[1];
			double cSizeX = cellSize[0];
			double cSizeY = cellSize[1];
		
			for(int y=0; y<sizeY; y++) {
				for(int x=0; x<sizeX; x++) {
					int cx = floor(x/cSizeX);
					int cy = floor(y/cSizeY);
				
					int num = 0;
					double val = img(y, x);
					double count = 0;
				
					for(int i=0; i<points; i++) {
						int tx = x + xpoints(i);
						int ty = y + ypoints(i);
						bool tmp = 0;
						if(tx>=0 && tx<sizeX && ty>=0 && ty<sizeY)
							if(val>img(ty, tx))
								tmp = 1;
						num = (num<<1) | tmp;
					
					}
				
					res(cy, cx, num) += 1;
				}
			}
		
			"""
		
			weave.inline(code, ['size', 'cellSize', 'radius', 'points', 'xpoints', 'ypoints', 'img', 'res'], type_converters=weave.converters.blitz, compiler='gcc')
		else:

			for y in range(size[0]):
				for x in range(size[1]):
					cx = floor(x/cellSize[0])
					cy = floor(y/cellSize[1])
					
					num = ''
					val = img[y, x]
					
					for i in range(0, points):
						tx = int(x + xpoints[i])
						ty = int(y + ypoints[i])
						
						if tx>=0 and tx<size[1] and ty>=0 and ty<size[0]:
							if val<img[ty, tx]:
								num += '1'
							else:
								num += '0'
						else:
							num += '0'
					
					num = int(num, 2)
					res[cy, cx, num] += 1 

		return res.ravel()	
		
	def update(self, image):
		self.X.append(self.__compute(image))
		
		
	def query(self, image):
		if len(self.X)==0:
			return .000001
		if 	id(image) not in self.cache:
			self.cache[id(image)] = numpy.cast[float](numpy.matrix(self.__compute(image)))
		img = self.cache[id(image)]
		
		res = []
		for i in range(0, len(self.X)):
			res.append(numpy.linalg.norm(self.X[i]-img))
			
		return numpy.average(res)

