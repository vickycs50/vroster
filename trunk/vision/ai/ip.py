import pymprog as mp
import numpy


class IP:
	
	def __init__(self):
		pass
		
		
	def simplify(self, c):
		res = c.copy()
		maxVal = 1e5
		minVal = 0
		
		# Restrict number of valid recognizers per object
		for i in range(c.shape[0]):
			best = numpy.argsort(c[i,:])
			if c.shape[0]<c.shape[1]:
				res[i, best[0,c.shape[0]:c.shape[1]]] = maxVal
		
		# Ignozer objects with high scores
		for i in range(c.shape[0]):
			if numpy.average(c[i,:])>70:
				res[i,:] = minVal
				
		return res
		
	def predict(self, c):
		#c = self.simplify(c)
		
		unknownCost = numpy.average(c, 1)
		c = numpy.hstack((c, unknownCost))

		added = None
		if c.shape[1]-c.shape[0]>1:
			added = numpy.zeros((c.shape[1]-c.shape[0], c.shape[1]))
			c = numpy.vstack((c, added))

		detected = c.shape[0]
		recognizers = c.shape[1]
		elems = detected * recognizers

		c = c.reshape((1, elems)).tolist()[0]

		A = []
		B = []

		# One label per detected object
		for i in range(0, detected):
			tmp = numpy.zeros((detected, recognizers))
			tmp[i, :] = 1
			tmp = tmp.reshape((1, elems))[0]
			A.append(tmp.tolist())
			B.append(1)
			
		# Use recognizer up to once
		for i in range(0, recognizers-1):
			tmp = numpy.zeros((detected, recognizers))
			tmp[:, i] = 1
			tmp = tmp.reshape((1, elems))[0]
			A.append(tmp.tolist())
			B.append(1)

		mp.beginModel('basic')
		mp.verbose(False)
		x = mp.var(range(elems), 'X', kind=bool)
		mp.minimize(
			sum(c[i]*x[i] for i in range(elems)), 'myobj'
		)
		r=mp.st(
			sum(x[j]*A[i][j] for j in range(elems)) == B[i] for i in range(len(A))
		)
		mp.solve(int)
		
		if added!=None:
			elems = elems - added.shape[0]*added.shape[1]
		X = numpy.zeros((1, elems))
		for i in range(elems):
			X[0, i] = x[i].primal
		return X.reshape((elems/(recognizers), recognizers))