import os
import cv
import sys
import numpy

from scikits.learn import svm
from scikits.learn.linear_model import SGDClassifier
from scikits.learn.naive_bayes import GNB

from ..util import Image
from BaseRecognizer import *

class BagRecognizer(BaseRecognizer):
	
	def __init__(self, directory, classifiers, size):
		self.classifiers = classifiers
		self.dist = None
		
		if getattr(directory, '__iter__', False) == False:
			directory = (directory, )

		cid = 0
		
		for currentDir in directory:
			directories = os.listdir(currentDir)
			list.sort(directories)
			
			for d in directories:
				if d[0]!='.':
				
					#print 'Loading recognizer [%02d]: %s\t%s'%(cid, currentDir, d)
					files = os.listdir(currentDir+'/'+d)
				
					if cid < len(self.classifiers):
						for f in files:
							if f[0] != '.' and f != 'info.txt':
								image = cv.LoadImage(currentDir+'/'+d+'/'+f, cv.CV_LOAD_IMAGE_GRAYSCALE)
							
								image2 = cv.CreateImage(size, image.depth, 1)
								cv.Resize(image, image2)
								
								x = Image.cv2array(image2)/255.0
								self.classifiers[cid].update(x)
							if f == 'info.txt':
								self.classifiers[cid].info =  open(currentDir+'/'+d+'/'+f, 'r').read()
					

					cid += 1
		for i in range(cid, len(self.classifiers)):
			self.classifiers.pop()

	def query(self, image):
		res = []
		
		for c in self.classifiers:
			res.append(c.query(image))

		return res
		
	def distance(self):
		if self.dist == None:
			self.dist = numpy.zeros((len(self.classifiers), len(self.classifiers)))
			for i in range(0, len(self.classifiers)):
				for j in range(0, len(self.classifiers)):
					a = numpy.average(self.classifiers[i].X, axis=0)
					b = numpy.average(self.classifiers[j].X, axis=0)
					self.dist[i,j] = numpy.linalg.norm(a-b)
		return self.dist
		
		
class BagRecognizerFDA(BagRecognizer):
	
	def __init__(self, directory, classifiers, size):
		BagRecognizer.__init__(self, directory, classifiers, size)
		
		training_data = []
		training_labels = []
		for (i,c) in enumerate(self.classifiers):
			for j in range(0, len(c.X)):
				training_data.append(c.X[j])
				training_labels.append(i)
		

		training_data = numpy.vstack(training_data)
		training_labels = numpy.hstack(training_labels)
		
		self.svm = []
		for i in range(0, len(self.classifiers)):
			#s = svm.SVC(kernel='rbf', probability=True)
			s = svm.LinearSVC()
			
			a = numpy.where(training_labels==i)[0]
			b = numpy.where(training_labels!=i)[0]
			
			l = training_labels.copy()
			l[a] = 1.0
			l[b] = 0.0
			
			print 'Fitting SVM ', i
			s.fit(training_data, l)

			self.svm.append(s)
			
	def query(self, image):
		test = self.classifiers[0].compute(image).tolist()
		test = numpy.matrix([test])
		
		res = []
		for i in range(0, len(self.classifiers)):
			#s = self.svm[i].predict_log_proba(test)
			s = self.svm[i].predict(test)[0]
			res.append(1-s)
		
		return res
		
	# 
	# 	print 'Building classifier'
	# 	#self.svm = svm.LinearSVC(multi_class=True)#SGDClassifier(n_iter=1000)
	# 	self.svm = svm.SVC(kernel='rbf', probability=True)
	# 	self.svm.fit(training_data, training_labels)
	# 	
	# 	
	# def query(self, image):
	# 	test = self.classifiers[0].compute(image).tolist()
	# 	test = numpy.matrix([test])
	# 	
	# 	res = []
	# 	for i in range(0, len(self.classifiers)):
	# 		s = self.svm.score(test, i*1.0)
	# 		#print self.svm.predict_proba(test), s
	# 		res.append(100-s)
	# 	res = 1-self.svm.predict_proba(test)[0]
	# 		
	# 
	# 	return res
	# 	
	# 	# best = self.svm.predict(test)
	# 	# res = []
	# 	# for i in range(0, len(self.classifiers)):
	# 	# 	res.append(self.classifiers[i].query(image))
	# 	# 	if i==best:
	# 	# 		res[-1] = res[-1]*.8
	# 	# return res
	# 	
	# 	