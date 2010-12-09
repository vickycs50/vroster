import numpy


class GapApproximation:
	def __init__(self):
		pass
		
	def predict(self, c):
		if c.size == 0:
			return []
		#print numpy.cast[int](c)	
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
			#print j, p, s
			t[int(s)] = j

		res = [-1 for i in range(0, c.shape[1])]
		for i in range(0, len(t)):
			if t[i]>-1:
				res[t[i]] = i
		return res
		
		
#c = numpy.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])*-1
#c = numpy.matrix([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])*-1
#print GapApproximation().predict(c)