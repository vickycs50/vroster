import numpy
import itertools
import scipy.spatial.distance as distance

class GapApproximation:
	def __init__(self):
		pass

	def predict(self, c):
		if c.size == 0:
		        return []
    
		c = ((c.max() + 1) - c).transpose()

		t = [-1 for i in range(0, c.shape[0])]


		for j in range(0, c.shape[1]):
		        p = []
		        for i in range(0, c.shape[0]):
		                if t[i]==-1:
		                        p.append(c[i, j])
		                else:
		                        p.append(c[i, j] - c[i, t[i]])

		        s = numpy.argmax(p)

		        t[int(s)] = j

		res = [-1 for i in range(0, c.shape[1])]
		for i in range(0, len(t)):
		        if t[i]>-1:
		                res[t[i]] = i
		return res		

		
class GapApproximation2(GapApproximation):
	
	def __init__(self, alpha, beta):
		self.prev = None
		self.diff = []
		self.alpha = alpha
		self.beta = beta
		
	def predict(self, c, desc = None):

		w = numpy.zeros((c.shape[0], c.shape[0]))
		
		if True and self.prev != None:
			for x in itertools.product(range(0, c.shape[0]), repeat=2):
				if x[0]<len(self.prev[1]) and x[1]<len(self.prev[1]):
					w[x[0], x[1]] = distance.euclidean(desc[x[0]], self.prev[1][x[1]])
		
		
			for i in range(0, c.shape[0]):
				for j in range(0, c.shape[1]):
					if i<len(self.prev[0]) and self.prev[0][i] == j:
						c[i,j] = c[i,j] - w[i,i]*self.alpha
		
		a = numpy.zeros((c.shape[0], 1))+numpy.average(c)*self.beta
		c = numpy.hstack((c,a))		
		
		p = GapApproximation.predict(self, c)
		
		if self.prev != None:
			d = 0
			for i in range(0, min(len(p), len(self.prev[0]))):
				if p[i]!=self.prev[0][i] or p[i]==-1:
					d += 1
			self.diff.append(d*1.0/len(p))
		
		self.prev = [p, desc]
		
		return p
		