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

class VRoster:
	
	def __init__(self, alpha=0, beta=0, priorSize=0, frames=300, skipFrames=5, ui=False):
		self.alpha = alpha 
		self.beta = beta
		self.priorSize = priorSize
		self.frames = frames
		self.skipFrames = skipFrames
		self.ui = ui
		
	def run(self):
		prev = []

		# Components
		video = CVVideo.CVFileVideo(config.TrialMovie)
		detector = SkinHaarDetector(config.HaarCascade, config.HaarSize, config.HaarSkin)
		tracker = TrivialTracker()
		ui = CVWindow(config.UIName, config.UISaveTo)

		if config.MinProblem == 'IP':
			ai = IP()
		elif config.MinProblem == 'Gap2':
			ai = GapApproximation()
		else:
			raise 'Minimization method not specified!'
			
		# Recognizer 
		recognizers = []
		for i in range(0, config.PhotoBag):
			recognizers.append(LBPRecognizer(cversion=True))
		recognizers = BagRecognizer(config.PhotoPath, recognizers, config.BoundingBox)

		track = []

		for fid in range(0, self.frames):
			#print fid, '%.02f%%'%(fid/(self.frames*1.0))
	
			for i in range(0, max(self.skipFrames, 1)):
				frame = video.next()
		
			if frame==None:
				print 'Movie ended!'
				break

			frameGray = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
			cv.CvtColor(frame, frameGray, cv.CV_RGB2GRAY)
       
			# Get objects
			observations = detector.detect(frame)
			objects = tracker.update(observations)
			objectImages = Image.extractSubImages(frameGray, objects, config.BoundingBox)
       
			# Generate recognition matrix
			recognized = []
			for image in objectImages:
				recognized.append(recognizers.query(image))
       
			# Attempt to find best matching
			w = numpy.matrix(recognized)
	
			prior = numpy.zeros(w.shape)
			priorHistory = numpy.zeros(w.shape)
	
			# Calculate total weights 
			if len(prev)>0 and self.priorSize>0:
				dist = recognizers.distance()
				
				for p in range(0, min(len(prev), self.priorSize)):
					pid = -1*p
					
					#print prev[pid][0]
					
					for o in range(0, w.shape[0]):
						if o < len(prev[pid][0]):
							for r in range(0, w.shape[1]):
								if prev[pid][0][o] != r:
									priorHistory[o, r] += 1
							prior[o, :] += dist[prev[pid][0][o], :]
				
				#w2 = w.copy()
				for o in range(0, w.shape[0]):
					w[o, :] += priorHistory[o, :]/max(numpy.sum(priorHistory[o, :]),1.0)*prior[o,:]*self.alpha
				#print numpy.cast[int](w-w2)
				#print priorHistory/self.priorSize
				
						
			#if len(prev)>0 and self.priorSize>0:
			#	for p in range(0, min(len(prev), self.priorSize)):
			#		pid = -1*p
			#		#for x in itertools.product(range(0, w.shape[0]), repeat=2):
			#		#	if x[0]<len(prev[pid][1]) and x[1]<len(prev[pid][1]):
			#		#		prior[x[0], x[1]] += numpy.linalg.norm(objectImages[x[0]]-prev[pid][1][x[1]])
			#		for x in range(0, priorHistory.shape[0]):
			#			for y in range(0, priorHistory.shape[1]):
			#				if y>=len(prev[pid][0]) or (x<len(prev[pid][0]) and prev[pid][0][x] != y):
			#					priorHistory[x, y] += 1
			#	for x in range(priorHistory.shape[0]):
			#		priorHistory[x,:] /= numpy.sum(priorHistory[x,:])
			#		#prior[x, :] /= min(len(prev), self.priorSize)*1.0
		    #
			#	prior = recognizers.distance()
			#	
			#	print prior.shape
			#	print priorHistory.shape
			#	
			#	for i in range(0, w.shape[0]):
			#		w[i,:] += prior[i]
			#	
			#	#for i in range(0, w.shape[0]):
			#	#	for j in range(0, w.shape[1]):
			#	#		if j<prior.shape[1]:
			#	#			w[i, j] += prior[i,j]*priorHistory[i,j]*self.alpha
		
			a = numpy.zeros((w.shape[0], 1)) + numpy.average(w)*self.beta
			w = numpy.hstack((w,a))
			print numpy.cast[int](w)
	
			predicted = ai.predict(w)
			prev.append([predicted, objectImages])

			
			currentTrack = []
			for (i,l) in enumerate(predicted):
				tmp = [objects[i][0]+objects[i][2]/2.0, objects[i][1]+objects[i][3]/2.0, l]
				currentTrack.append(tmp)

			#track.append(predicted)
			track.append(currentTrack)	
			
			if self.ui == True:
				canvas = CVCanvas(frame)
       
				if len(predicted)>0:
					for i in range(0, len(objects)):
						label = predicted[i]
						canvas.drawText('[%02d]=%d'%(i, label), (objects[i][0], objects[i][1]-3), (255, 255, 0))
						canvas.drawRect(objects[i], (255,0,0))
			
				ui.update(canvas)

		return track
		
if __name__ == '__main__':
	v = VRoster(alpha=1, beta=1, priorSize=5, ui=True)
	v.run()		