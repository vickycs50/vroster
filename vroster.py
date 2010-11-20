import pickle

from vision.util import CVVideo, Image
from vision.detector import *
from vision.tracker import *
from vision.recognizer.LBPRecognizer import *
from vision.recognizer.BagRecognizer import *
from vision.matlab.interface import *
from vision.ui.CVInterface import *
from vision.ai.trivial import *

# UI settings
ui = CVWindow('VRoster')
	
# Video settings
videoID = 2
videoPath = '/Users/andre/Desktop/videoroster/%03d.mov'%videoID

# Photo settings
photoPath = '/Users/andre/Desktop/videoroster/photos2/'

# Detector settings
cascadePath = 'data/opencv-24x24.xml'
cascadeSize = (12,12)

# Recognition
boundSize = (24, 24)

# Matlab
matlab = LocalMatlab(10, '/Applications/MATLAB_R2010b.app/', 'maci64')
matlab.addpath('matlab/')
matlab.execExpression('addpath(genpath(\'matlab/yalmip\'))', verbose=True)

# Components
video = CVVideo.CVFileVideo(videoPath)
detector = HaarDetector(cascadePath, cascadeSize)
tracker = TrivialTracker()
ai = dict()

# Recognizer 
recognizers = []
for i in range(0, 8):
	recognizers.append(LBPRecognizer(matlab))
recognizers = BagRecognizer(photoPath, recognizers, boundSize)

while True:
	frame = video.next()
	if frame==None:
		break
	frameGray = Image.toGray(frame)
	
	# Get objects
	observations = detector.detect(frameGray)
	objects = tracker.update(observations)
	objectImages = Image.extractSubImages(frameGray, objects, boundSize)
	
	# Generate recognition matrix
	recognized = []
	for image in objectImages:
		recognized.append(recognizers.query(image))

	# Attempt to find best matching
	w = numpy.matrix(recognized)
	res = matlab.vroster_ipo(w, 2, verbose=True)
	labels = numpy.matrix(res[0])
	
	predicted = []
	for i in range(0, labels.shape[0]):
		if i not in ai:
			#ai[i] = MultinomialAI(range(0,9))
			ai[i] = TrivialAI()
		ai[i].update(numpy.argmax(labels[i,:]))
		predicted.append(ai[i].predict())
			
	
	
	if ui != None:
		canvas = CVCanvas(frame)
		
		for i in range(0, len(objects)):
			label = predicted[i][0]
			conf = predicted[i][1]
						
			canvas.drawText('%d ~ %.02f'%(label, conf), (objects[i][0], objects[i][1]-3), (255, 255, 0))
			canvas.drawRect(objects[i], (255,0,0))
			
		
		cv.ShowImage('VRoster', frame)	
		ui.update(canvas)
	
	#if cv.WaitKey(1)!=-1:
	#	break
