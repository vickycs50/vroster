import cv

from BaseDetector import *


class HaarDetector(BaseDetector):
	"""OpenCV Haar detector implementation"""
	
	def __init__(self, cascade, size):
		self.cascade = cv.Load(cascade)
		self.size = size
		
	def detect(self, image):
		haarResults = cv.HaarDetectObjects(image, self.cascade, cv.CreateMemStorage(0), 1.2, 1, cv.CV_HAAR_DO_CANNY_PRUNING, self.size)

		res = []
		for r in haarResults:
			a = list(r[0])
			a[0] -= 10
			a[1] -= 10
			a[2] += 10
			a[3] += 10
			res.append(a)
			
		return res
		
class FastHaarDetector(HaarDetector):
	
	def __init__(self, cascade, size, mod=[2,2]):
		HaarDetector.__init__(self, cascade, size)
		self.mod = mod
		self.ind = [0, 0]
		
	def detect(self, image):
		size = cv.GetSize(image)
		ss = (size[0]/self.mod[0], size[1]/self.mod[1])
		currSlice = (ss[0]*self.ind[0], ss[1]*self.ind[1], ss[0], ss[1])

		self.ind[0] += 1
		if self.ind[0]==self.mod[0]:
			self.ind[0] = 0
			self.ind[1] +=1
		if self.ind[1]==self.mod[1]:
			self.ind[0] = 0
			self.ind[1] = 0
			
		img = cv.GetSubRect(image, currSlice)
		res = HaarDetector.detect(self, img)
		
		for i in range(0, len(res)):
			res[i] = [res[i][0]+currSlice[0], res[i][1]+currSlice[1], res[i][2], res[i][3]]
			
		return res
		
class SkinHaarDetector(HaarDetector):
	
	def __init__(self, cascade, size, skin):
		HaarDetector.__init__(self, cascade, size)
		self.skin = skin
		
		self.h = None
		self.s = None
		self.v = None
		self.resA = None
		self.resB = None
		self.resC = None
		
		if skin['Debug'] == True:
			cv.NamedWindow('SkinHaarDebug', 1)

	
	def __getObjects(self, seq):
		res = []
		while seq:
			res.append(cv.BoundingRect(list(seq)))
			res.extend(self.__getObjects(seq.v_next()))
			seq = seq.h_next()
		return res
	
	def getSkin(self): 
		return self.res
	def getSkinPercent(self):
		return self.skinPerc
		
	def detect(self, image):
		size = cv.GetSize(image)
		hsv = cv.CloneImage(image)
		
		if self.h == None:
			self.h = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
			self.s = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
			self.v = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
			self.resA = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
			self.resB = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
			self.resC = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
		
		cv.CvtColor(image, hsv, cv.CV_RGB2HSV)
		cv.Split(hsv, self.h, self.s, self.v, None)
		
		self.res = cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)
		
		cv.CmpS(self.h, self.skin['HMin'], self.resA, cv.CV_CMP_GT)
		cv.CmpS(self.h, self.skin['HMax'], self.resB, cv.CV_CMP_LT)
		cv.And(self.resA, self.resB, self.res)
		cv.CmpS(self.s, self.skin['SMin'], self.resA, cv.CV_CMP_GT)
		cv.CmpS(self.s, self.skin['SMax'], self.resB, cv.CV_CMP_LT)
		cv.And(self.resA, self.resB, self.resC)
		cv.And(self.resC, self.res, self.res)
		cv.CmpS(self.v, self.skin['VMin'], self.resA, cv.CV_CMP_GT)
		cv.CmpS(self.v, self.skin['VMax'], self.resB, cv.CV_CMP_LT)
		cv.And(self.resA, self.resB, self.resC)
		cv.And(self.resC, self.res, self.res)
		
		cv.Dilate(self.res, self.res, None, self.skin['Dilate'])
		cv.Erode(self.res, self.res, None, self.skin['Erode'])

		if self.skin['Debug'] == True:
			cv.ShowImage('SkinHaarDebug', self.res)

		storage = cv.CreateMemStorage()
		contour = cv.CloneImage(self.res)
		objects = cv.FindContours(contour, storage, cv.CV_RETR_TREE, cv.CV_CHAIN_APPROX_SIMPLE)
		objects = self.__getObjects(objects)

		detected = []
		self.skinPerc = 0
		
		for obj in objects:
			if obj[2]>self.skin['MinSize'] and obj[3]>self.skin['MinSize']:
				cv.ResetImageROI(image)
				cv.SetImageROI(image, obj)
				found = HaarDetector.detect(self, image)
				for f in found:
					f[0] = f[0] + obj[0]
					f[1] = f[1] + obj[1]
					if f[2]<self.skin['MaxSize'] and f[3]<self.skin['MaxSize']:
						self.skinPerc += f[2]*f[3]
						detected.append(f)

		self.skinPerc /= float(size[0]*size[1])
		cv.ResetImageROI(image)
		
		return detected
		