import numpy

class TrivialAI:
	
	def update(self, state):
		self.state = state
		
	def predict(self):
		return [self.state, 0]
		
class MultinomialAI:
	
	def __init__(self, states):
		self.states = numpy.matrix(states)*1.0
		
	def update(self, state):
		self.states[0, state] += 1
		
	def predict(self):
		w = self.states[0,:]/(numpy.sum(self.states))
		s = numpy.random.multinomial(100, w.tolist()[0], 1)
		l = numpy.argmax(s)
		return [l, self.states[0,l]/numpy.sum(self.states)]