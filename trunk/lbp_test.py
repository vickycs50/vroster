import numpy
import vision.recognizer.LBPRecognizer as lbp
import vision.util.Profile as profile

a = lbp.LBPRecognizer()
p = profile.Profile()


for n in range(0, 25):
	p.start('Init')
	a.update(numpy.random.rand(100,100))
	p.end('Init')


for n in range(0, 100):
	p.start('Test')
	a.query(numpy.random.rand(100,100))
	p.end('Test')


p.stats()