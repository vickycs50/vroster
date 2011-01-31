import pymprog as mp
import numpy
import scipy.stats as stats


class IP:
	
	def __init__(self):
		pass
		
	def predict(self, c):
		if c.size == 0:
			return []

		added = None
		addedElems = 0
		if c.shape[1]-c.shape[0]>1:
			added = numpy.zeros((c.shape[1]-c.shape[0], c.shape[1]))
			addedElems = added.shape[0] * added.shape[1]
			c = numpy.vstack((c, added))

		detected = c.shape[0]
		recognizers = c.shape[1]
		elems = detected * recognizers
		
		mp.beginModel('basic')
		mp.verbose(False)
		x = mp.var(range(elems), 'X', kind=bool)

		# One label per detected object
		for i in range(0, detected):
			tmp = numpy.zeros((detected, recognizers))
			tmp[i, :] = 1
			tmp = tmp.reshape((1, elems))[0]
			mp.st(sum(x[j]*int(tmp[j]) for j in range(elems))==1)
			
		# Use recognizer up to once
		for i in range(0, recognizers-1):
			tmp = numpy.zeros((detected, recognizers))
			tmp[:, i] = 1
			tmp = tmp.reshape((1, elems))[0]
			mp.st(sum(x[j]*int(tmp[j]) for j in range(elems))==1)
			
		c = c.reshape((1, elems)).tolist()[0]
			
		mp.minimize(sum(c[i]*x[i] for i in range(elems)), 'myobj')
		mp.solve(int)
		
		X = numpy.zeros((1, elems))
		for i in range(elems):
			X[0, i] = x[i].primal
		X = X[0,0:elems-addedElems]
		
		labels = X.reshape(((elems-addedElems)/(recognizers), recognizers))
		predicted = []
		for i in range(0, labels.shape[0]):
			l = numpy.argmax(labels[i,:])
			if l==recognizers-1:
				l = -1
			predicted.append(l)
			
		return predicted