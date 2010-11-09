import cv
import numpy
import scipy.spatial.distance as distance
import sys
import util

imageAPath = '/Users/andre/Desktop/videoroster/photos2/01/02.png'
imageBPath = '/Users/andre/Desktop/videoroster/photos2/02/05.png'


imageA = cv.LoadImage(imageAPath, cv.CV_LOAD_IMAGE_GRAYSCALE)
imageB = cv.LoadImage(imageBPath, cv.CV_LOAD_IMAGE_GRAYSCALE)

a = util.cv2array(imageA)/255.0 * 255
b = cv.fromarray(numpy.cast['uint8'](a))
print numpy.asarray(b)

(pointsA, descA) = cv.ExtractSURF(b, None, cv.CreateMemStorage(), (1, 500, 3, 4))
(pointsB, descB) = cv.ExtractSURF(imageB, None, cv.CreateMemStorage(), (1, 500, 3, 4))

dist = numpy.zeros((len(descA), len(descB)))

for m in range(0, len(descA)):
	for n in range(0, len(descB)):
		dist[m, n] = distance.euclidean(descA[m], descB[n])	
	
print numpy.average(numpy.min(dist, axis=1))