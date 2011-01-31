import cv
import math
import time

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .35, .35)

class CVWindow:
	
	def __init__(self, name=None, output=None):
		
		self.name = None
		self.out = None
		
		if name!=None:
			self.name = name
			cv.NamedWindow(name, 1)
			
		if output!=None:
			self.out = cv.CreateVideoWriter(output, cv.CV_FOURCC('P','I','M','1'), 30, (1280, 720), True)
			
		
	def update(self, canvas):
	
		self.canvas = canvas
		self.curr = canvas.resize(1)
		
		if self.name!=None:
			cv.ShowImage(self.name, self.curr)
			cv.WaitKey(1)
		if self.out!=None:
			cv.WriteFrame(self.out, self.curr)
		


class CVCanvas:
	
	def __init__(self, img):
		self.image = cv.CloneImage(img)
		
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
		
	def saveWithID(self, folder, fid):
		cv.SaveImage('%s/%04d.png'%(folder, fid), self.resize(1))