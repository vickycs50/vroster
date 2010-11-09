import cv
import numpy
import util
import sys
import sampling

class Logic:
	
	def assign(self, classifiers, detectedPos, detectedImages):
		pass
		
class RandomLogic(Logic):
	
	def __init__(self):
		self.posterior = dict()
		
		pass
	
	def assign(self, classifiers, detectedPos, detectedImg):
		if len(detectedPos)==0:
			return []
		
		likelihood = numpy.zeros(shape=(len(classifiers), len(detectedImg)))
		for c in range(0, likelihood.shape[0]):
			for o in range(0, likelihood.shape[1]):
				s = classifiers[c].query(detectedImg[o])
				likelihood[c, o] = s
				
		
		#(labels, objects) = self.__gibs(classifiers, likelihood)	
		(labels, objects) = self.__trivial(likelihood)
		
		for i in range(0, len(labels)):
			#s = score[labels[i], objects[i]]
			s = 1
			if objects[i] in self.posterior:
				self.posterior[objects[i]][labels[i]] += s
			else:
				self.posterior[objects[i]] = numpy.zeros((len(classifiers), 1))
				self.posterior[objects[i]][labels[i]] += s
		
		
		scores = []
		for i in range(0, len(labels)):
			
			w = likelihood[:,objects[i]]*self.posterior[objects[i]]
			w = w/w.sum()
			
			scores.append(likelihood[labels[i], objects[i]]/likelihood[:, objects[i]].sum())
			#labels[i] = sampling.weighedSample(w.T[0])
			#scores.append(w[labels[i]][0])

		print likelihood
		print scores
		print objects, labels
			
		#for i in range(0, len(labels)):
		#	if labels[i]>=0:
		#		classifiers[labels[i]].update(detectedImg[objects[i]], scores[i])

		results = []
		for i in range(0, len(labels)):
			r = dict()
			if labels[i]>=0:
				r['score'] = scores[i]
			else:
				r['score'] = 0
			r['label'] = labels[i]
			r['rect'] = detectedPos[objects[i]]
			results.append(r)

		return results
	
	def __combination(self, labels, size):
		res = []
		for i in range(0, size):
			currLabels = labels[:]
			for n in res:
				currLabels.remove(n)
			r = numpy.random.random_integers(0, len(currLabels), 1)
			res.extend([currLabels[r]])
		return res
			
	def __gibs(self, classifiers, weights):
		labels = []
		objects = []
		
		(o, tmp) = sampling.GibbsSampler(weights, 500)
		c = range(0, len(classifiers))
        
		order = numpy.argsort(weights[c, o])
		
		for i in order:
			if o[i] not in objects:
				objects.append(o[i])
				labels.append(c[i])
		for i in range(0, weights.shape[1]):
			if i not in objects:
				objects.append(i)
				labels.append(-1)		
		
		return (labels, objects)
		
	def __trivial(self, weights):
		labels = []
		objects = []
		
		for i in range(0, weights.shape[1]):
			if weights[:,i].sum()==0:
				labels.append(-1)
				objects.append(i)
			else:
				labels.append(numpy.argmax(weights[:,i]))
				objects.append(i)
			
		return (labels, objects)