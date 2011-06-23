import numpy
import itertools
import scipy.spatial.distance as distance


## Basic GAP approximation alogirthm
class GapApproximation:
	def __init__(self):
		pass

	## c -- cost matrix to perform paring on
	## unknown -- flag telling if the unknown label is part of the cost matrix c
	def predict(self, c, unknown):
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
		if unknown==True:
			lim = len(t)-1
		else:
			lim = len(t)
			
		for i in range(0, lim):
			if t[i]>-1:
					res[t[i]] = i
		return res		

