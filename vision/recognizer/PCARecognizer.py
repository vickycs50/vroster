import numpy
import mdp.nodes as mdp

from BaseRecognizer import *

## Attempt to use PCA as a recognizer... did not work
class PCA(BaseRecognizer):
	
	def __init__(self, data):
		self.X = data
		self.pca = None
		self.__update()
		
	def update(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))

		self.X = numpy.vstack((self.X, data))
		self.__update();
		
	def __update(self):
		self.pca = mdp.PCANode(output_dim=.95)
		self.pca.train(self.X)
		
	def query(self, data):
		if data.shape[0] != 1:
			data = numpy.reshape(data, (1, data.size))
	
		return self.pca(data)