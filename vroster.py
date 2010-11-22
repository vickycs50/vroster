import pickle

import config
from vision.util import CVVideo, Image
from vision.detector import *
from vision.tracker import *
from vision.recognizer.LBPRecognizer import *
from vision.recognizer.BagRecognizer import *
from vision.matlab.interface import *
from vision.ui.CVInterface import *
from vision.ai.trivial import *

# Setup UI
if config.EnableUI == True:
	ui = CVWindow('VRoster')
else:
	ui = None

# Matlab
matlab = LocalMatlab(config.MatlabVersion, config.MatlabPath, config.MatlabArch)
matlab.addpath('matlab/')
matlab.execExpression('addpath(genpath(\'matlab/yalmip\'))')

# Components
video = CVVideo.CVFileVideo(config.TrialMovie)
detector = HaarDetector(config.HaarCascade, config.HaarSize)
tracker = TrivialTracker()
ai = dict()

# Recognizer 
recognizers = []
for i in range(0, 8+5):
	recognizers.append(LBPRecognizer(matlab))
recognizers = BagRecognizer(config.PhotoPath, recognizers, config.BoundingBox)

while True:
	frame = video.next()
	if frame==None:
		break
	frameGray = Image.toGray(frame)
	
	# Get objects
	observations = detector.detect(frameGray)
	objects = tracker.update(observations)
	objectImages = Image.extractSubImages(frameGray, objects, config.BoundingBox)
	
	# Generate recognition matrix
	recognized = []
	for image in objectImages:
		recognized.append(recognizers.query(image))

	# Attempt to find best matching
	w = numpy.matrix(recognized)
	print w
	res = matlab.vroster_ipo(w, 2)
	print res
	sys.exit(0)
	labels = numpy.matrix(res[0])
	
	# Extract labels for each object
	predicted = []
	for i in range(0, labels.shape[0]):
		predicted.append(numpy.argmax(labels[i,:]))
	
	print predicted		

	if ui != None:
		canvas = CVCanvas(frame)
		
		for i in range(0, len(objects)):
			label = predicted[i][0]
			conf = predicted[i][1]
						
			canvas.drawText('%d ~ %.02f'%(label, conf), (objects[i][0], objects[i][1]-3), (255, 255, 0))
			canvas.drawRect(objects[i], (255,0,0))
			
		
		cv.ShowImage('VRoster', frame)	
		ui.update(canvas)

