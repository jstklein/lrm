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
import matplotlib.pyplot as plt

# File-related Settings
logFileName='logfile.txt'
baseDir = './data/'
experimentDir='test'
experimentDataPath = baseDir + experimentDir
bfFileName = 'beta'

# Capture settings
betaGain = 2.5
betaShutterS = 10
betaSecondsPerImage = 60*1

betaShutterUs = betaShutterS * 1000 * 1000
numImagesToSum = int(betaSecondsPerImage/betaShutterS)

# Initialize LRM class
LRM = LRM()
  
# Get exposure settings for brightfield images and lock them in
bfGain,bfShutter = LRM.get_brightfield_exposure()

# Override brightfield settings
bfShutter = 6000
bfGain = 1
threshold=3

print("Starting acquisition of one " + str(betaSecondsPerImage) + "s beta image")

# Capture brightfield 
LRM.capture_brightfield(experimentDataPath, 'bf-0', 1, bfGain, bfShutter)
	
# Capture multiple beta images and integrate into one file
LRM.capture_beta(experimentDataPath, 'beta-0', numImagesToSum, betaGain, betaShutterUs, threshold)

LRM.preview_last(True)

