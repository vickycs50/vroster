import numpy

class TrivialAI:
	
	def __init__(self):
		pass
		
	def predict(self, c):
		res = []
		
		for i in range(0, c.shape[0]):
			res.append(numpy.argmin(c[i,:]))
			
		return res