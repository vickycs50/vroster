import pickle
import traceback
import sys

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

# Setup UI
ui = None
if config.EnableUI == True:
	ui = CVWindow(None, None)

# Weight of past history
alpha = 0
# Weight of "unknown face"
beta = .100
# Previous solution
prev = None

# Components
video = CVVideo.CVFileVideo(config.TrialMovie)
detector = SkinHaarDetector(config.HaarCascade, config.HaarSize, config.HaarSkin)
tracker = TrivialTracker()
if config.MinProblem == 'IP':
	ai = IP()
elif config.MinProblem == 'Gap2':
	ai = GapApproximation()
else:
	raise 'Minimization method not specified!'
profile = Profile()

# Recognizer 
recognizers = []
for i in range(0, config.PhotoBag):
	recognizers.append(LBPRecognizer(cversion=False))
recognizers = BagRecognizer(config.PhotoPath, recognizers, config.BoundingBox)

track = []


try:
	for fid in range(0, 180):
		print fid, '%.02f%%'%(fid/180.0)
		
		profile.start('FPS')
		frame = video.next()
		if frame==None:
			print 'Movie ended!'
			break

		frameGray = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
		cv.CvtColor(frame, frameGray, cv.CV_RGB2GRAY)

		# Get objects
		profile.start('Haar')
		observations = detector.detect(frame)
		profile.end('Haar')

		profile.start('Track')
		objects = tracker.update(observations)
		profile.end('Track')

		objectImages = Image.extractSubImages(frameGray, objects, config.BoundingBox)

		# Generate recognition matrix
		recognized = []
		profile.start('LBP')
		for image in objectImages:
			recognized.append(recognizers.query(image))
		profile.end('LBP')

		# Attempt to find best matching
		w = numpy.matrix(recognized)
		
		prior = numpy.zeros((w.shape[0], w.shape[0]))
		
		if alpha!=0 and prev != None:
			for x in itertools.product(range(0, w.shape[0]), repeat=2):
				if x[0]<len(prev[1]) and x[1]<len(prev[1]):
					prior[x[0], x[1]] = distance.euclidean(objectImages[x[0]], prev[1][x[1]])

			for i in range(0, len(prev[0])):
				if prev[0][i]>-1:
					w[i, prev[0][i]] -= prior[i,i]*alpha
				else:
					w[i, prev[0][i]] += prior[i,i]*alpha
			
		a = numpy.zeros((w.shape[0], 1)) + numpy.average(w)*beta
		w = numpy.hstack((w,a))
		
		
		profile.start('IP')
		predicted = ai.predict(w)
		profile.end('IP')
		
		print numpy.cast[int](w)
		print predicted
		
		prev = [predicted, objectImages]
		
		track.append(predicted)

		if ui != None:
			canvas = CVCanvas(frame)

			if len(predicted)>0:
				for i in range(0, len(objects)):					
					label = predicted[i]
					canvas.drawText('%d'%(label), (objects[i][0], objects[i][1]-3), (255, 255, 0))
					canvas.drawRect(objects[i], (255,0,0))
				
			#ui.update(canvas)
			#canvas.saveWithID('gap-10-08', fid)
			
		profile.end('FPS')
except KeyboardInterrupt:
	print ''
	profile.stats()
except Exception as e:
	traceback.print_exc()
