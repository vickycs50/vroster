import cv
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


def toGray(image):
	imageGray = cv.CreateImage(cv.GetSize(image), 8, 1)
	cv.CvtColor(image, imageGray, cv.CV_RGB2GRAY)
	cv.EqualizeHist(imageGray, imageGray)
	return imageGray
	
def extractSubImage(image, region, size):
	if region[0]<0:
		region = (0, region[1], region[2], region[3])
	if region[1]<0:
		region = (region[0], 0, region[2], region[3])

	sub = cv.GetSubRect(image, region)
	box = cv.CreateImage(size, 8, 1)
	cv.Resize(sub, box)
	return cv2array(box)/255.0
	
def extractSubImages(image, regions, size):
	res = []
	
	for region in regions:
		try:
			res.append(extractSubImage(image, region, size))
		except Exception as e:
			print e
	return res