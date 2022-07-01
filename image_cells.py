'''image_cells.py

Protocol for designed for long duration brightfield and beta imaging of living cells

The protocol loops numberCaptures times
During each loop it:
	captures 1 brightfield image
	captures 1 beta image of betaSecondsPerImage total duration time

rev 2
Justin Klein
Stanford University
Department of Radiation Oncology
2018
'''
from lrm import LRM
import time


# File-related Settings
baseDir = './data/'
experimentDir='10-min-0Gy-fdg-5MBq-dynamic-0g'
experimentDataPath = baseDir + experimentDir
betaPath = experimentDataPath + '/beta'
bfPath = experimentDataPath + '/bf-png' 

# Capture settings
betaGain = 2.5
betaShutterS = 10
betaSecondsPerImage = 60*10 #300s = 5 minutes
numberCaptures = 12
# 5 minutes x 12 images = 60 minutes
threshold=3

betaShutterUs = betaShutterS * 1000 * 1000
numImagesToSum = int(betaSecondsPerImage/betaShutterS)

# Initialize LRM class
LRM = LRM()
  
# Get exposure settings for brightfield images and lock them in
bfGain, bfShutter = LRM.get_brightfield_exposure()

# Override brightfield settings
bfShutter = 6000
bfGain = 1


print(f"Starting acquisition of {numberCaptures} {betaSecondsPerImage} second beta images")


for n in range(numberCaptures):

	# Capture brightfield 
	LRM.capture_brightfield(experimentDataPath, 'bf-' + str(n+startOffset) + '-', 1, bfGain, bfShutter)
	
	# Capture multiple beta images and integrate into one file
	LRM.capture_beta(experimentDataPath, 'beta-'+str(n+startOffset), numImagesToSum, betaGain, betaShutterUs, threshold)
	
	# Preview and don't halt for display
	LRM.preview_last(False)
