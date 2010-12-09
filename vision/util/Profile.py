import time
import numpy

class Profile:
	
	def __init__(self):
		self.data = dict()
		self.queue = dict()
		self.initTime = time.time()
		
	def start(self, k):
		if k in self.queue:
			self.queue[k].append(time.time())
		else:
			self.queue[k] = [time.time()]
		
	def end(self, k):
		start = self.queue[k].pop()
		
		if k in self.data:
			self.data[k].append(time.time()-start)
		else:
			self.data[k] = [time.time()-start]
			
			
	def stats(self):
		total = time.time() - self.initTime
		
		print '%-10s\t%-7s\t%-7s'%('Name', 'Avg', 'Perc')
		for key in self.data:
			avg = numpy.average(self.data[key])
			perc = numpy.sum(self.data[key])/total
			print '%-10s\t%3.3f\t%2.2f'%(key, avg, perc)
			