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
		self.recognizers = BagRecognizerFDA(config.PhotoPath, self.recognizers, config.BoundingBox)

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

		# Generate recognition matrix
		recognized = []
		for image in objectImages:
			recognized.append(self.recognizers.query(image))
   
		# Attempt to find best matching
		w = numpy.matrix(recognized)
		
		print w
		
		
		prior = numpy.zeros(w.shape)
		priorHistory = numpy.zeros(w.shape)

		# Calculate total weights 
		if len(self.prev)>0 and self.priorSize>0:
			if self.dist == None:
				self.dist = self.recognizers.distance()
			
			for p in range(0, min(len(self.prev), self.priorSize)):
				pid = -1*p
				
				for o in range(0, w.shape[0]):
					if o < len(self.prev[pid][0]):
						for r in range(0, w.shape[1]):
							if self.prev[pid][0][o] != r:
								priorHistory[o, r] += 1
						try:	
							prior[o, :] += self.dist[self.prev[pid][0][o], :]
						except Exception:
							pass
							#print 'Prior exception'
							
			for o in range(0, w.shape[0]):
				w[o, :] += priorHistory[o, :]/max(numpy.sum(priorHistory[o, :]),1.0)*prior[o,:]*self.alpha

		a = numpy.zeros((w.shape[0], 1)) + numpy.average(w,1)*self.beta - numpy.std(w,1)*.5
		w = numpy.hstack((w,a))
	
		predicted = self.ai.predict(w)
		#print w, predicted
		self.prev.append([predicted, objectImages])
		if len(self.prev)>self.priorSize:
			self.prev.pop()

		currentTrack = []
		if len(objects)>0:
			for (i,l) in enumerate(predicted):
				tmp = [objects[i][0]+objects[i][2]/2.0, objects[i][1]+objects[i][3]/2.0, l]
				currentTrack.append(tmp)
		#print >>self.output, currentTrack
		
		return (objects, predicted)


	
	def run(self, config=None):
		self.setup(config)

		for fid in range(0, self.frames):
			print fid, '%.02f%%'%(fid/(self.frames*1.0))
	
			for i in range(0, max(self.skipFrames, 1)):
				frame = self.video.next()
				
			if frame==None:
				print 'Movie ended!'
				break
			
			objects, predicted = self.update(frame)
 		
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
	
	v = VRoster(alpha=1.75, beta=1, priorSize=5, skipFrames=5, frames=4050, ui=True)
	track = v.run(config.VideoRoster())		

	
 