import time
import numpy

class Profile:
	
	def __init__(self):
		self.data = dict()
		self.queue = dict()
		self.initTime = time.clock()
		
	def start(self, k):
		if k in self.queue:
			self.queue[k].append(time.clock())
		else:
			self.queue[k] = [time.clock()]
		
	def end(self, k):
		start = self.queue[k].pop()
		
		if k in self.data:
			self.data[k].append(time.clock()-start)
		else:
			self.data[k] = [time.clock()-start]
			
			
	def stats(self):
		total = time.clock() - self.initTime
		
		print '%-10s\t%-7s\t%-7s'%('Name', 'Avg', 'Perc')
		for key in self.data:
			print '%-10s\t%3.3f\t%2.2f'%(key, numpy.average(self.data[key]), numpy.sum(self.data[key])/total)
			