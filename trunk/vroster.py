import cv
import sys
import numpy

import detector
import classifier
import util
import logic

for vidID in range(2, 41):

	# Settings
	moviePath = '/Users/andre/Desktop/videoroster/%03d.mov'%vidID
	picturePath = '/Users/andre/Desktop/videoroster/photos2/'
	cascadePath = 'data/opencv-24x24.xml'
	cascadeSize = (12,12)
	#cascadePath = '/opt/local/share/opencv/haarcascades/haarcascade_frontalface_alt_tree.xml'
	#cascadeSize = (15,15)
	#cascadePath = '/opt/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
	#cascadeSize = (24,24)
	useUI = False
	skipFrames = 1
	clipSize = (32,32)

	# Setup
	movie = cv.CaptureFromFile(moviePath)
	detector = detector.HaarDetector(cascadePath, cascadeSize)
	classifiers = util.createClassifiers(picturePath, classifier.Template, clipSize)
	logic = logic.RandomLogic()
	font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .4, .4)

	if useUI == True:
		cv.NamedWindow('VRoster')

	frames = []

	while cv.GrabFrame(movie)==1:
		frame = cv.QueryFrame(movie)
		frameGray = cv.CreateImage(cv.GetSize(frame), 8, 1)
		cv.CvtColor(frame, frameGray, cv.CV_RGB2GRAY)
		cv.EqualizeHist(frameGray, frameGray)
	
		# Actual work
		detectedPos = detector.detect(frameGray)
		detectedImg = util.extractRegions(frameGray, detectedPos, clipSize)
	
	
		table = numpy.zeros((len(classifiers), 35))
		for c in range(0, len(classifiers)):
			for i in range(0, len(detectedImg)):
				table[c, i] = classifiers[c].query(detectedImg[i])
		
		frames.append(frames)
		print vidID, len(frames)
	
		#results = logic.assign(classifiers, detectedPos, detectedImg)

		#for result in results:
		#	if result['score']!=0:
		#		blue = 255 * (1-result['score'])
		#		red = 255 * result['score']
	    #
		#		tl = (result['rect'][0], result['rect'][1])
		#		lr = (result['rect'][0]+result['rect'][2], result['rect'][1]+result['rect'][3])
		#		cv.Rectangle(frame, tl, lr, (red,0,blue))
		#	
		#		label = '%02d - %.02f'%(result['label'], result['score']) 
		#		cv.PutText(frame, label, (tl[0], tl[1]-2), font, (0, 255, 0))
		#
		## Display video
		#if useUI == True:
		#	cv.ShowImage('VRoster', frame)
	
		# Skip ahead a bit
		currentFrame = cv.GetCaptureProperty(movie, cv.CV_CAP_PROP_POS_FRAMES)
		cv.SetCaptureProperty(movie, cv.CV_CAP_PROP_POS_FRAMES, currentFrame+skipFrames)
	
	
		# Check to quit
		#if cv.WaitKey(0) == 'q':
		#	break;

	plicke.dump(frames, open('tracked-%03d.dat'%vidID, 'wb'))
	