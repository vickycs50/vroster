# 
#	Configuration for VRoster
#

# --------------------------------------------------------------------
# Trial settings
# --------------------------------------------------------------------
# Path for movie to use
TrialMovie = '/Users/andre/Desktop/videoroster/002.mov'
# Path of where pictures are located
# - Each person has a subdirectory with images inside, a recognizer is
#   made for each sub folder and loads all images inside that folder
PhotoPath = '/Users/andre/Desktop/videoroster/photos2/'

# --------------------------------------------------------------------
# VRoster parameters
# --------------------------------------------------------------------
# True - Displays progress with a window
EnableUI = False
# Recognized faces are resized to a fixed size
BoundingBox = (24, 24)

# --------------------------------------------------------------------
# Matlab 
# --------------------------------------------------------------------
MatlabVersion = 10
# Path to the location Matlab is stored
MatlabPath = '/Applications/MATLAB_R2010b.app/'
# Architecture matlab is compiled for
MatlabArch = 'maci64'

# --------------------------------------------------------------------
# Haar
# --------------------------------------------------------------------
HaarCascade = 'data/opencv-24x24.xml'
HaarSize = (24, 24)

