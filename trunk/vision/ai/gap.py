import numpy


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
			#print j, p, s
			t[int(s)] = j

		res = [-1 for i in range(0, c.shape[1])]
		for i in range(0, len(t)):
			if t[i]>-1:
				res[t[i]] = i
		return res
		
class GapApproximation2(GapApproximation):
	
	def __init__(self):
		self.prev = None
		
	def predict(self, c):
		p = GapApproximation.predict(self, c)

		if self.prev!=None:
			link = numpy.zeros(c.shape)
			for m in range(0, c.shape[0]):
				for n in range(0, c.shape[1]):
					try: 
						if self.prev[1][m] == n:
							link[m,n] = 0
						else:
							link[m,n] = abs(c[m,n]-self.prev[0][m, n])
					except:
						link[m,n] = c[m,n]
			p = GapApproximation.predict(self, c+5*link)
		self.prev = [c, p]
		
		return p
		
		
				
#c = numpy.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])*-1
#c = numpy.matrix([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])*-1
#print GapApproximation().predict(c)