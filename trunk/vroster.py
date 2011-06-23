import pickle
import traceback
import sys
import numpy

import config

from vision.util import CVVideo, Image
from vision.util.Profile import *
from vision.detector import *
from vision.tracker import *
from vision.recognizer.LBPRecognizer import *
from vision.recognizer.BagRecognizer import *
from vision.recognizer.BagRecognizerSVM import *
from vision.recognizer.DistRecognizer import *
from vision.ui.CVInterface import *
from vision.ai.ip import *
from vision.ai.gap import *
from vision.ai.trivial import *

class VRoster:
	
	def __init__(self, alpha=0, beta=0, priorSize=0, frames=300, skipFrames=5, ui=False, output=None):
		self.alpha = alpha 
		self.beta = beta
		self.priorSize = priorSize
		self.frames = frames
		self.skipFrames = skipFrames
		self.ui = ui
		self.profile = Profile()
		if output == None:
			self.output = open('output.txt', 'w+')
		else:
			self.output = output
	
	def setup(self, config=None):
		

		# Components
		self.video = None
		if config.TrialMovie != None:
			self.video = CVVideo.CVFileVideo(config.TrialMovie)
		self.detector = SkinHaarDetector(config.HaarCascade, config.HaarSize, config.HaarSkin)
		self.tracker = TrivialTracker(config)
		self.window = CVWindow(config.UIName, config.UISaveTo)

		# Determine minimization algorithm to be used
		if config.MinProblem == 'IP':
			self.ai = IP(constraints=config.MinProblemConstraints)
		elif config.MinProblem == 'Gap2':
			self.ai = GapApproximation()
		elif config.MinProblem == 'Trivial':
			self.ai  = TrivialAI()
		else:
			raise 'Minimization method not specified!'

		# Recognizer 
		self.recognizers = []
		for i in range(0, config.PhotoBag):
			self.recognizers.append(LBPRecognizer(cversion=True))
		self.recognizers = BagRecognizerSVM(config.PhotoPath, self.recognizers, config.BoundingBox)

		self.track = []
		self.prev = []
		self.config = config
		self.dist = None
	
	def update(self, frame):
		
		frameGray = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
		cv.CvtColor(frame, frameGray, cv.CV_RGB2GRAY)
   
		# Get objects
		observations = self.detector.detect(frame)
		objects = self.tracker.update(observations)
		objectImages = Image.extractSubImages(frameGray, objects, self.config.BoundingBox)
		
		detectedCount = len(objectImages)
		
		# Generate recognition matrix
		recognized = []
		for image in objectImages:
			recognized.append(self.recognizers.query(image))
   
		# Attempt to find best matching
		s = numpy.abs(numpy.matrix(recognized))

		# Temporal variables
		b = numpy.eye(detectedCount)
		da = numpy.zeros((detectedCount, detectedCount))
		ds = numpy.zeros(s.shape)
	
	 	# Calculate temporal variables
		if len(self.prev)>0:
			for t in range(0, len(self.prev)):
				for i in range(0, detectedCount):
					for j in range(0, len(self.prev[0][1])):					
						da[i,j] += numpy.sum(numpy.power(objectImages[i]-self.prev[t][1][j],2))**.5
			da /= (len(self.prev)*1.0)
		
		
			for i in range(0, ds.shape[0]):
				for j in range(0, ds.shape[1]):
					tmp = []
					for k in range(0, len(self.prev[0][0])):
						tmp2 = 0
						for t in range(0, len(self.prev)):
							tmp2 += (1 - 2*int(self.prev[t][0][k]==j))
						tmp.append(tmp2/(len(self.prev)*1.0)*b[i,k]*da[i,k])
					ds[i,j] = numpy.sum(tmp)
		
		
		# Create cost matrix
		s = s + self.beta*ds
		
		# Unknown column is added
		a = numpy.zeros((detectedCount, 1)) + numpy.average(s,1)*self.alpha - numpy.std(s,1)*.5
		s = numpy.hstack((s,a))
		
		# Solve the pairing problem (assuming B is correct)
		c = self.ai.predict(s, unknown=True)
		
		# Calculate new B
		if len(self.prev)>0:
			w = numpy.zeros((detectedCount, detectedCount))
			for i in range(0, detectedCount):
				for j in range(0, len(self.prev[0][0])):
					w[i,j] = da[i,j] * int(c[i]!=self.prev[0][0][j])
		
			# Solve the pairing problem for B (assuming C is correct)
			newB = self.ai.predict(w, unknown=False)
		
			# Update B matrix with results
			b = numpy.eye(detectedCount)
			for i in range(0, len(newB)):
				b[i, newB[i]] = 1

		# Update prior data
		self.prev.append([c, objectImages])
		if len(self.prev)>self.priorSize:
			self.prev.pop()

		# Save tracking information for display and output
		currentTrack = []
		if len(objects)>0:
			for (i,l) in enumerate(c):
				tmp = [objects[i][0]+objects[i][2]/2.0, objects[i][1]+objects[i][3]/2.0, l]
				currentTrack.append(tmp)
				
				self.output.write('%d %d %d; '%(tmp[0], tmp[1], tmp[2]))
		self.output.write('\n')
		
		return (objects, c)


	# Main code that cycles through frames performing the recognition and tracking
	def run(self, config=None):
		self.setup(config)

		for fid in range(0, self.frames):
			print fid, '%.02f%%'%(fid/(self.frames*1.0))
	
			for i in range(0, max(self.skipFrames, 1)):
				frame = self.video.next()
				
			if frame==None:
				print 'Movie ended!'
				break
			
			# Track/Recognize
			objects, predicted = self.update(frame)
 		
			# If UI is enabled, display results
			if self.ui == True:
				canvas = CVCanvas(frame)
				
				if len(objects)>0:
					for i in range(0, len(objects)):
						label = predicted[i]
						canvas.drawText('[%02d]=%d'%(i, label), (objects[i][0], objects[i][1]-3), (255, 255, 0))
						canvas.drawRect(objects[i], (255,0,0))
			
				self.window.update(canvas)

		return track
		
if __name__ == '__main__':
	
	v = VRoster(alpha=1, beta=1, priorSize=5, skipFrames=2, frames=4050, ui=True)
	
	track = v.run(config.VideoRoster())		
	#track = v.run(config.LargeClassroom())		
	
 