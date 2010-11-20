import cv

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .35, .35)

class CVWindow:
	
	def __init__(self, name):
		self.name = name
		cv.NamedWindow(name, 0)
		
	def update(self, canvas):
		cv.ShowImage(self.name, canvas.image)


class CVCanvas:
	
	def __init__(self, image):
		self.image = cv.CloneImage(image)
		
	def drawRect(self, rect, color):
		tl = (rect[0], rect[1])
		lr = (rect[0]+rect[2], rect[1]+rect[3])
		cv.Rectangle(self.image, tl, lr, color)
		
	def drawText(self, text, pos, color):
		cv.PutText(self.image, text, pos, font, color)