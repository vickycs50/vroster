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
#from vision.matlab.interface import *
from vision.ui.CVInterface import *
from vision.ai.ip import *
from vision.ai.gap import *

# Setup UI
ui = None
if config.EnableUI == True:
	ui = CVWindow('VRoster')

# Components
video = CVVideo.CVFileVideo(config.TrialMovie)
detector = SkinHaarDetector(config.HaarCascade, config.HaarSize, config.HaarSkin)
tracker = TrivialTracker()
ai = GapApproximation()
profile = Profile()

# Recognizer 
recognizers = []
for i in range(0, config.PhotoBag):
	recognizers.append(LBPRecognizer())
recognizers = BagRecognizer(config.PhotoPath, recognizers, config.BoundingBox)

try:
	#while True:
	for i in range(0,250):
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
		objects = tracker.update(observations)
		
		objectImages = Image.extractSubImages(frameGray, objects, config.BoundingBox)
	
		# Generate recognition matrix
		recognized = []
		profile.start('LBP')
		for image in objectImages:
			recognized.append(recognizers.query(image))
		profile.end('LBP')
		
		# Attempt to find best matching
		w = numpy.matrix(recognized)
		profile.start('IP')
		predicted = ai.predict(w)
		profile.end('IP')
	
		if ui != None:
			canvas = CVCanvas(frame)

			if len(predicted)>0:
				
				avgDist = []
				print predicted
				for i in range(0,len(predicted)):
					if predicted[i]>0:
						avgDist.append(w[i, predicted[i]])
				avgDist = numpy.average(avgDist)
				
				for i in range(len(objects)):
					label = predicted[i]
					conf = avgDist/w[i, label]
						
					canvas.drawText('%d ~ %.02f'%(label, conf), (objects[i][0], objects[i][1]-3), (255, 255, 0))
					canvas.drawRect(objects[i], (255,0,0))
			
			ui.update(canvas)
			
			#cv.WaitKey(-1)
		profile.end('FPS')
except KeyboardInterrupt:
	print ''
	profile.stats()
	sys.exit(0)
except Exception as e:
	traceback.print_exc()
	sys.exit(0)
profile.stats()
