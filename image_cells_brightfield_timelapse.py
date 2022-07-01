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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# File-related Settings
baseDir = './data/'
experimentDir='testtesttest'
experimentDataPath = baseDir + experimentDir

numberCaptures = 1000
timeBetweenCapturesSeconds = 60*1

# Initialize LRM class
LRM = LRM()
  
# Get exposure settings for brightfield images and lock them in
bfGain,bfShutter = LRM.get_brightfield_exposure()
bfShutter = 5000

print("Starting acquisition of " + str(numberCaptures) + " brightfield images")


#fig = plt.figure()
for n in range(numberCaptures):

	# Announce
	print("Image " + str(n+1) + "/" + str(numberCaptures) )

	# Capture brightfield 
	LRM.capture_brightfield(experimentDataPath, 'bf-' + str(n+1) + '-', 1, bfGain, bfShutter)
	
	# Display
	LRM.preview_last()
	
	# Sleep
	print("Sleeping for " + str(timeBetweenCapturesSeconds) + "s")
	plt.pause(timeBetweenCapturesSeconds)
	#time.sleep(timeBetweenCapturesSeconds)
