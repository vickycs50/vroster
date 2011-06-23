
## Base configuration class
class Config:
	# Specifies Haar settings
	HaarCascade = 'data/opencv-24x24.xml'
	HaarSize = (24,24)
	BoundingBox = (24, 24)
	
	# Name of window to show results (set to None if no display is wanted)
	UIName = "Vroster"
	# Name of folder to save results to (image results, not tracking data)
	UISaveTo = None
	
	# Tunes skin finding algorithm, the base settings basically disables the functionality
	HaarSkin = {
		# Debug=True if you want to see what it thinks skin was
		'Debug': False,
		'HMin': 0,
		'HMax': 256,
		'SMin': 0,
		'SMax': 256,
		'VMin': 0,
		'VMax': 256,
		'Dilate': 1,
		'Erode': 1, 
		'MinSize': 30,
		'MaxSize': 100
	}
	
	# Specifies the assignment algorithm to be used
	#MinProblem = 'IP'
	MinProblem = 'Gap2'
	#MinProblem = 'Trivial'

	# Should use every identity only once
	MinProblemConstraints = True
	# Should do temporal tracking
	TrackerEnabled = True
	TrackerDistance = 100
	
class VideoRoster(Config):	
	TrialMovie = '/Users/andre/Datasets/videoroster/clips/002.mov'
	PhotoPath = ['/Users/andre/Datasets/videoroster/photos-class/']
	PhotoBag = 8
	
	# Testing with more pictures/known faces
	#PhotoPath = ['/Users/andre/Datasets/videoroster/photos-class/', '/Users/andre/Datasets/videoroster/lecture-faces-2/']
	#PhotoBag = 16
	
	
	HaarSkin = {
		'Debug': False,
		'HMin': 50,
		'HMax': 230,
		'SMin': 80,
		'SMax': 150,
		'VMin': 0,
		'VMax': 256,
		'Dilate': 3,
		'Erode': 1, 
		'MinSize': 30,
		'MaxSize': 100
	}

	# Path to file to be analyzed
	TrialMovie = None
	# Path to face database (in form of a list)
	PhotoPath = [None]
	# Number of identities to populate database with
	PhotoBag = None

class LargeClassroom(Config):
	TrialMovie = '/Users/andre/Datasets/videoroster/large-classroom2.mov'
	PhotoPath = ['/Users/andre/Datasets/videoroster/lecture-faces-2/']
	PhotoBag = 28
	
	HaarSkin = {
		'Debug': False,
		'HMin': 50,
		'HMax': 230,
		'SMin': 80,
		'SMax': 150,
		'VMin': 0,
		'VMax': 256,
		'Dilate': 3,
		'Erode': 1, 
		'MinSize': 30,
		'MaxSize': 100
	}

class YoutubeVideo(Config):
	
	TrialMovie = '/Users/andre/Datasets/worldvideo/youtube.mov'
	PhotoPath = ['/Users/andre/Datasets/worldvideo/face/']
	PhotoBag = 1
