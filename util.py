import cv
import os
import sys
import numpy

def cv2array(im):
	depth2dtype = {
		cv.IPL_DEPTH_8U: 'uint8',
		cv.IPL_DEPTH_8S: 'int8',
		cv.IPL_DEPTH_16U: 'uint16',
		cv.IPL_DEPTH_16S: 'int16',
		cv.IPL_DEPTH_32S: 'int32',
		cv.IPL_DEPTH_32F: 'float32',
		cv.IPL_DEPTH_64F: 'float64',
	}

	arrdtype=im.depth
	a = numpy.fromstring(
		im.tostring(),
		dtype=depth2dtype[im.depth],
		count=im.width*im.height*im.nChannels)
	a.shape = (im.height,im.width,im.nChannels)
	return a[:,:,0]*1.0
	
def array2cv(a):
	dtype2depth = {
		'uint8':   cv.IPL_DEPTH_8U,
		'int8':    cv.IPL_DEPTH_8S,
		'uint16':  cv.IPL_DEPTH_16U,
		'int16':   cv.IPL_DEPTH_16S,
		'int32':   cv.IPL_DEPTH_32S,
		'float32': cv.IPL_DEPTH_32F,
		'float64': cv.IPL_DEPTH_64F,
	}
	try:
		nChannels = a.shape[2]
	except:
		nChannels = 1
	cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
		dtype2depth[str(a.dtype)],
		nChannels)
	cv.SetData(cv_im, a.tostring(),
		a.dtype.itemsize*nChannels*a.shape[1])
	return cv_im
	
	
def createClassifiers(directory, type, size):
	directories = os.listdir(directory)
	
	res = []
	
	for d in directories:
		if d[0]!='.':
			print 'Loading:',d 
			files = os.listdir(directory+'/'+d)
			X = []
			for file in files:
				if file[0] != '.':
					image = cv.LoadImage(directory+'/'+d+'/'+file, cv.CV_LOAD_IMAGE_GRAYSCALE)
					image2 = cv.CreateImage(size, image.depth, 1)
					cv.Resize(image, image2)
					x = cv2array(image2)/255.0
					x = numpy.reshape(x, (1, x.size))
					X.append(x)
			
			X = numpy.hstack(X)
			X = numpy.reshape(X, (len(files), X.size/len(files)))
			res.append(type(X))
		
	return res
	
def extractRegionsCV(image, region, size):
	cv.SetImageROI(image, region)
	region = cv.CreateImage(cv.GetSize(image), image.depth, image.nChannels)
	cv.Copy(image, region)
	region2 = cv.CreateImage(size, region.depth, region.nChannels)
	cv.Resize(region, region2)
	cv.ResetImageROI(image)
	
	return region2	
	
def extractRegions(image, regions, size):
	res = []
	
	for region in regions:
		try:
			sub = cv.GetSubRect(image, region)
			box = cv.CreateImage(size, 8, 1)
			cv.Resize(sub, box)
			res.append(cv2array(box)/255.0)
		except Exception as e:
			print e
	return res
