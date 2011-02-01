import numpy
import vision.recognizer.LBPRecognizer as lbp
import vision.util.Profile as profile

a0 = lbp.LBPRecognizer(cversion=True)
a1 = lbp.LBPRecognizer(cversion=False)

numpy.random.seed(0)
b = numpy.cast[int](numpy.random.rand(10,10)*100)

a0.update(b)
a1.update(b)

print b
print numpy.sum(a0.X[0] - a1.X[0])