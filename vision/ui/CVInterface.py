import cv
import math
import time

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .35, .35)

class CVWindow:
	
	def __init__(self, name):
		self.name = name
		cv.NamedWindow(name, 1)
		
	def update(self, canvas):
		self.canvas = canvas
		self.curr = canvas.resize(1)
		cv.ShowImage(self.name, self.curr)
		cv.WaitKey(1)


class CVCanvas:
	
	def __init__(self, img):
		self.image = cv.CloneImage(img)
		#cv.ResetImageROI(self.image)
		
	def drawRect(self, rect, color):
		tl = (rect[0], rect[1])
		lr = (rect[0]+rect[2], rect[1]+rect[3])
		cv.Rectangle(self.image, tl, lr, color)
		
	def drawText(self, text, pos, color):
		cv.PutText(self.image, text, pos, font, color)
		
	def resize(self, fraction):
		s = cv.GetSize(self.image)
		c = cv.CreateImage((int(s[0]*fraction), int(s[1]*fraction)), self.image.depth, self.image.nChannels)
		cv.Resize(self.image, c, cv.CV_INTER_CUBIC)
		return c