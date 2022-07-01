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
experimentDir='one-brightfield'
experimentDataPath = baseDir + experimentDir
bfFileName = 'bf-0'

# Initialize LRM class
LRM = LRM()
  
# Get exposure settings for brightfield images and lock them in
bfGain,bfShutter = LRM.get_brightfield_exposure()

# Override brightfield shutter duration if so desired
bfShutter=10000

# Announce
print("Capturing single brightfield image. Shutter = " + str(bfShutter) + ' ms')

# Capture brightfield 
LRM.capture_brightfield(experimentDataPath,bfFileName, 1, bfGain, bfShutter)
LRM.preview_last(True)
