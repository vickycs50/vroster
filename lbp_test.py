import numpy
import vision.recognizer.LBPRecognizer as lbp
import vision.util.Profile as profile

a0 = lbp.LBPRecognizer(cversion=True)
a1 = lbp.LBPRecognizer(cversion=False)

b = numpy.random.rand(32,32)

a0.update(b)
a1.update(b)


print a0.X
print a1.X