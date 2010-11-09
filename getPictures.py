import cv
import os
import random 
import util

def mouse(event, x, y, flags, param):
	pos = None
	if event==cv.CV_EVENT_LBUTTONUP:
		pos = (x, y)
		
		if param[0] == None:
			param[0] = pos
		else:
			poi = (param[0][0], param[0][1], pos[0]-param[0][0], pos[1]-param[0][1])

			img = util.extractRegionsCV(currentImage, poi)
			cv.SaveImage(param[1]+'/'+param[2]+'/%02d'%(param[3])+'.png', img)
			
			param[3] = param[3] + 1
			param[0] = None
			
			print 'OK'

storePath = '/Users/andre/Desktop/videoroster/photos2/'
videoPath = '/Users/andre/Desktop/videoroster/CBIM_Classroom_Video.avi'
person = '08'
video = cv.CaptureFromFile(videoPath)
currentImage = None

params = [None, storePath, person, 0]

cv.NamedWindow('Video', cv.CV_WINDOW_AUTOSIZE)
cv.SetMouseCallback('Video', mouse, params)

for n in range(0, 10):
	maxFrames = cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FRAME_COUNT)
	nextFrame = random.uniform(100, maxFrames)

	cv.SetCaptureProperty(video, cv.CV_CAP_PROP_POS_FRAMES, nextFrame)
	currentImage = cv.QueryFrame(video)
	
	cv.ShowImage('Video', currentImage)
	
	cv.WaitKey(0)
	

