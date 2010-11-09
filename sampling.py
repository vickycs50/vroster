import numpy
import sys
import math


def weighedSample(w):
	if numpy.sum(w[:-1])!=1:
		w = w/numpy.sum(w)

	s = numpy.random.multinomial(1, w, 1)

	return numpy.where(s==1)[1][0]
	
def generateSample(weights):
	res = []
	for n in range(0, weights.shape[0]):
		res.append(weighedSample(weights[n,:]))
	return res
	
def rejectionSample(weights, target, iter=500):
	count = 0
	for i in range(0, iter):
		s = generateSample(weights)
		if s==target:
			count += 1
	print count
	return count/(iter*1.0)
	
def sampleScore(weights, config, i=None):
		
	p1 = weights[range(0, len(config)), config].prod()

	if i!=None:
		p2 = weights[i[0], i[1]]
	else:
		p2 = 1
		
	return p1/p2

def GibbsSampler(weights, iter=100):

	bestConfig = generateSample(weights)
	bestConfigScore = sampleScore(weights, bestConfig)+1e-09

	for i in range(0, iter):

		config = list(bestConfig)	

		for n in range(0, weights.shape[0]):
			config[n] = weighedSample(weights[n,:])
		
		n = numpy.random.randint(0, weights.shape[0])
		config[n] = weighedSample(weights[n,:])

		score = sampleScore(weights, config, (n, config[n]))+1e-09

		#if 1<score/bestConfigScore:
		if numpy.random.rand() < score/bestConfigScore:
			print 'Old', bestConfig, bestConfigScore
			bestConfig = config
			bestConfigScore = score
			print 'New', bestConfig, bestConfigScore

	return (bestConfig, bestConfigScore)

if __name__ == "__main__":
	weights = numpy.matrix([[1,1,1,2],
							[2,1,2,1],
							[1,2,1,2]])*1.0
						
	weights = weights / numpy.sum(weights,axis=1)


	print GibbsSampler(weights)		
