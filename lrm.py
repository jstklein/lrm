''' LRM.py
Defines LRM class that contains all beta microscope functionality
Justin Klein
Stanford University
Department of Radiation Oncology
2018
'''

import picamera
import io
from PIL import Image
import numpy as np
import time
import datetime
import os
import glob
from gpiozero import LED
import pickle
import lz4.frame
from lrmimage import LrmImage

# mpl packages
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''CONSTANTS'''
LED_PIN = 21

'''V2 resolutions (3280,2464) (1920,1080) (1640,1232) (1640,922) (1280,720)  (640,480)'''
DEFAULT_RESOLUTION = (3280, 2464)
DEFAULT_SENSOR_MODE = 3
DEFAULT_THROW_AWAY_FRAMES = 1
DEFAULT_FRAMERATE_RANGE = (0.1, 15)
DEFAULT_AWB_GAINS = (1, 1)

CAMERA_TIMEOUT = 120

SHUTTER_THRESHOLD = 10
GAIN_THRESHOLD = 25
BETA_IMAGE_BITS = 16
BETA_IMAGE_THRESHOLD = 0

'''SET CAPTURE TIMEOUT'''
picamera.PiCamera.CAPTURE_TIMEOUT = CAMERA_TIMEOUT


# Some useful functions
def is_within_percent(a, b, percent):
    """ Return true if a is percent within b """
    diff = abs(a - b)

    if a == 0:
        p = (diff) * 100

    else:
        p = (diff / a) * 100.

    within = p < percent
    return within


def append_slash(path):
    """ Append a slash a path if one doesn't already exist """
    if not (path[-1] == '/'):
        path += '/'

    return path


def get_parent_directory(path):
    """ Get the parent directory of a given path """
    isFilePath = path[-4] == '.' and not (path[-1] == '/')

    if isFilePath:
        directory = path[: path.rfind('/') + 1]
        return directory
    else:
        return append_slash(path)


def check_or_make_directory(path):
    """ Check for or make path to data sub directory 
    """

    path = get_parent_directory(path)
    directory = os.path.dirname(path)
    dirExists = os.path.exists(directory)

    if not dirExists:
        os.makedirs(directory)


class LRM:
    """Class containing all beta microscope functionality

    Example use case:
    
    Capture brightfield images as separate files:
        
        gain, shutter = LRM.get_brightfield_exposure()
        
        lrm = LRM()    
        
        lrm.snap_brightfield('./data/bf/','bf', 1, gain, shutter)

    Capture multiple beta images as separate files:
        lrm.snap_betas('./data/beta/','bm_5g_10000ms',1000,2.5,10)

    Capture and sum multiple beta image into a single file single file:
        lrm.snap_beta('./data/beta/','bm_5g_10000ms_1000fr',1000,2.5,10)
    
    """

    def __init__(self):
        self.lastSaveFileFullPath = None
        self.lastCaptureDurationSeconds = None
        self.logFileFullPath = None
        self.led = None
        self.lastBfImage = None
        self.lastBetaImage = None

        self._gainSet = None
        self._shutterSet = None

        if self.led == None:
            self.led = LED(LED_PIN)
            self.led.off()
        self.info = {}

    def __setup_beta(self, camera, gain=None, shutterUs=None):
        """Start up the camera with settings optimized for beta imaging
        gain: analog gain
        shutteUs: shutter speed in microseconds
        """

        # Gain and shutter speed must be provided
        if gain is None or shutterUs is None:
            return False

        # Copy provided values into LRM gain and shutter setpoint variables
        self._gainSet = gain
        self._shutterSet = shutterUs

        # Settings for beta imaging mode
        camera.framerate_range = (0.1, 15)
        camera.resolution = (3280, 2464)
        camera.sensor_mode = 3
        camera.sharpness = 0
        camera.contrast = 0
        camera.brightness = 50
        camera.saturation = 0
        camera.video_stabilization = False
        camera.exposure_compensation = 0
        camera.meter_mode = 'average'
        camera.image_effect = 'none'
        camera.image_denoise = False
        camera.color_effects = None
        camera.drc_strength = 'off'
        camera.awb_gains = (1, 1)
        camera.awb_mode = 'auto'
        camera.exposure_mode = 'auto'

        # Set the gain
        camera.exposure_mode = 'auto'

        if gain < 2.5:
            camera.shutter_speed = 500 * 1000
            camera.iso = 60
            time.sleep(2)

        if (gain >= 2.5) and (gain < 5):
            camera.shutter_speed = 100 * 1000
            camera.iso = 150
            time.sleep(2)

        if (gain >= 5) and (gain < 7.5):
            camera.shutter_speed = 50 * 1000
            camera.iso = 300
            time.sleep(3)

        if (gain >= 7.5) and (gain < 10):
            camera.shutter_speed = 15 * 1000
            camera.iso = 475
            time.sleep(3)

        if gain >= 10:
            camera.shutter_speed = 8 * 100
            camera.iso = 600
            time.sleep(3)

        # Set the shutter speed
        camera.shutter_speed = shutterUs

        framerate = 1. / (shutterUs / 1000000.)

        if not (framerate > 60 or framerate < 0.1):
            camera.framerate = framerate

        camera.awb_mode = 'off'
        camera.exposure_mode = 'off'

        # Make sure the camera is setup correctly
        # self.__check_camera(camera)

        # wait for automatic gain/shutter adjustment
        time.sleep(2)

    def __setup_brightfield(self, camera, gain=None, shutterUs=None):
        """Start up the camera with settings optimized for beta imaging
        gain: analog gain
        shutteUs: shutter speed in microseconds
        """

        # Gain and shutter speed must be provided
        if gain is None or shutterUs is None:
            return False

        # Copy provided values into LRM gain and shutter setpoint variables
        self._gainSet = gain
        self._shutterSet = shutterUs

        # Settings for brightfield imaging mode
        camera.framerate_range = (30, 30)
        camera.resolution = (3280, 2464)
        camera.sensor_mode = 3
        camera.sharpness = 0
        camera.contrast = 0
        camera.brightness = 50
        camera.saturation = 0
        camera.video_stabilization = False
        camera.exposure_compensation = 0
        camera.meter_mode = 'average'
        camera.image_effect = 'none'
        camera.image_denoise = False
        camera.color_effects = None
        camera.drc_strength = 'off'
        camera.awb_gains = (1, 2)
        camera.awb_mode = 'auto'

        # Set the gain
        if gain < 2.5:
            camera.shutter_speed = 500 * 1000
            camera.iso = 60
            time.sleep(2)

        if (gain >= 2.5) and (gain < 5):
            camera.shutter_speed = 100 * 1000
            camera.iso = 150
            time.sleep(2)

        if (gain >= 5) and (gain < 7.5):
            camera.shutter_speed = 50 * 1000
            camera.iso = 300
            time.sleep(3)

        if (gain >= 7.5) and (gain < 10):
            camera.shutter_speed = 15 * 1000
            camera.iso = 475
            time.sleep(3)

        if gain >= 10:
            camera.shutter_speed = 8 * 100
            camera.iso = 600
            time.sleep(3)

        # Set the shutter speed
        camera.shutter_speed = shutterUs

        framerate = 1. / (shutterUs / 1000000.)

        if not (framerate > 60 or framerate < 0.1):
            camera.framerate = framerate

        camera.awb_mode = 'off'
        camera.exposure_mode = 'off'

        # Make sure the camera is setup correctly
        # self.__check_camera(camera)

        # wait for automatic gain/shutter adjustment
        # time.sleep(2)

    def __check_camera(self, camera):
        """ Check both the gain and exposure settings
        """
        return (self.__check_shutter(camera) and self.__check_gain(camera))

    def __check_shutter(self, camera):
        """ Compare camera shutter speed with the desired setpoint
        returns: True if the shutter speed is within SHUTTER_THRESHOLD of the shutter setpoint
        """

        # Sometimes the camera is still initializing and shutter_speed
        # doesn't have a value; in that case wait till it has something
        while camera.shutter_speed is None:
            time.sleep(0.1)

        if self._shutterSet is None:
            return True
        else:
            shutter_us = camera.shutter_speed
            exposure_us = camera.exposure_speed

        shutterCorrect = is_within_percent(shutter_us, self._shutterSet, SHUTTER_THRESHOLD)

        return shutterCorrect

    def __check_gain(self, camera):
        """ Compare camera analog gain with the desired setpoint
        returns: True if the analog gain is within GAIN_THRESHOLD of the analog gain setpoint
        """

        analog_gain = camera.analog_gain

        while analog_gain == 0:
            print("analog gain = 0 ... retrying")

            time.sleep(5)

            stream = io.BytesIO()
            next(camera.capture_continuous(stream, 'jpeg', use_video_port=True, bayer=False))

            analog_gain = camera.analog_gain
            # give camera time to

        if self._gainSet is None:
            gainCorrect = True
        else:
            gainCorrect = is_within_percent(analog_gain, self._gainSet, GAIN_THRESHOLD)

        return gainCorrect

    def __reboot(self, camera):
        """ Reboots the camera
        """

        # Framerate needs to be set to 1 in order to let the camera close at long shutter speeds
        print("Setting framerate to 1 and closing camera ...")
        camera.framerate = 1
        camera.close()
        print("..done.")

        camera = picamera.PiCamera()
        time.sleep(3)

        return camera

    def __stream_to_np(self, stream):
        """Convert a supplied stream into an numpy array
        returns image from stream as numpy array
        """

        stream.seek(0)

        # Copy locally
        streamBuffer = stream.getvalue()

        # Convert to numpy array
        image = np.array(Image.open(io.BytesIO(streamBuffer)))

        # Convert to B/W by summing color channels into one 16-bit channel
        if len(image.shape) == 3:
            image = image.sum(axis=2, dtype=np.uint16)

        return image

    def __stream_to_RGB_np(self, stream):
        """Convert a supplied stream into an numpy array
        """

        stream.seek(0)

        # Copy locally
        streamBuffer = stream.getvalue()

        # Convert to numpy array
        image = np.array(Image.open(io.BytesIO(streamBuffer)))
        # image = image.astype(np.uint8)

        return image

    def __get_info(self, camera):
        """ Grab settings from the camera and return as dictionary
        """

        info = {'analog_gain': float(camera.analog_gain),
                'framerate_range': (float(camera.framerate_range[0]), float(camera.framerate_range[1])),
                'sharpness': camera.sharpness,
                'brightness': camera.brightness,
                'saturation': camera.saturation,
                'video_stabilization': camera.video_stabilization,
                'exposure_compensation': camera.exposure_compensation,
                'meter_mode': camera.meter_mode,
                'image_effect': camera.image_effect,
                'image_denoise': camera.image_denoise,
                'color_effects': camera.color_effects,
                'drc_strength': camera.drc_strength,
                'awb_gains': (float(camera.awb_gains[0]), float(camera.awb_gains[1])),
                'iso': camera.iso,
                'shutter_speed': camera.shutter_speed,
                'exposure_speed': camera.exposure_speed,
                'awb_mode': camera.awb_mode,
                'exposure_mode': camera.exposure_mode,
                'sensor_mode': camera.sensor_mode,
                'resolution': (camera.resolution[0], camera.resolution[1]),
                'datetime': datetime.datetime.now}

        return info

    def __snap_beta(self, camera, threshold=0):
        """Grab frame from camera and return stream as numpy array
        rgb: returns image as 3d numpy array with RGB channels
        returns: numpy array of camera image
        """

        # Check, reboot, and setup until camera settings are correct
        while not self.__check_camera(camera):
            print("Check camera failed, rebooting ...")
            print(f"Gain setpoint = {self._gainSet} Current = {camera.analog_gain}")
            print(f"Shutter setpoint = {self._shutterSet} Current = {camera.shutter_speed}")

            camera = self.__reboot(camera)
            self.__setup_beta(camera, self._gainSet, self._shutterSet)

        # Capture from the camera
        stream = io.BytesIO()
        captureStartTime = time.time()
        next(camera.capture_continuous(stream, 'jpeg', use_video_port=True, bayer=False))
        captureTimeSeconds = time.time() - captureStartTime

        # Grab image from camera (returned as stream)
        self.info = self.__get_info(camera)
        self.info['capture_time_s'] = captureTimeSeconds

        if stream is not None:
            # Copy covert stream to np array and copy into image variable
            image = self.__stream_to_np(stream)

            # Apply threshold
            image[image < threshold] = 0

            # Compute some image metrics and store in info
            self.info['image_sum'] = image.sum()
            self.info['image_std'] = np.std(image)
            self.info['image_max'] = image.max()
            self.info['image_std'] = image.min()
            self.info['image_mean'] = image.mean()

            return image, self.info

    def __snap_brightfield(self, camera):
        """Grab frame from camera and return stream as numpy array
        rgb: returns image as 3d numpy array with RGB channels
        returns: numpy array of camera image
        """

        # Check, reboot, and setup until camera settings are correct
        while not self.__check_camera(camera):
            print("Check camera failed, rebooting ...")
            print("Gain setpoint = " + str(self._gainSet) + " Current = " + str(camera.analog_gain))
            print("Shutter setpoint = " + str(self._shutterSet) + " Current = " + str(camera.shutter_speed))

            camera = self.__reboot(camera)
            self.__setup_brightfield(camera, self._gainSet, self._shutterSet)

        # Capture from the camera
        stream = io.BytesIO()
        captureStartTime = time.time()
        next(camera.capture_continuous(stream, 'jpeg', use_video_port=True, bayer=False))
        captureTimeSeconds = time.time() - captureStartTime

        # Grab image from camera (returned as stream)
        self.info = self.__get_info(camera)
        self.info['capture_time_s'] = captureTimeSeconds

        if stream is not None:
            image = self.__stream_to_RGB_np(stream)

            return image, self.info

    def capture_betas(self, fullPath, filePrefix, numberImages, analogGain, exposureDurationUs):
        """Capture desired number of frames from camera object and write to disk (save threaded version)
        fullPath: save path
        filePrefix: prefix of saved file
        numberImages: number of frames to capture
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure the LED is off just in case+
        self.led.off()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        # Set up camera for beta imaging mode
        self.__setup_beta(camera, analogGain, exposureDurationUs)

        # Announce
        print(f"Capturing beta image(s): path = {fullPath} prefix = {filePrefix} # {numberImages}\n"
              f"Gain = {camera.analog_gain} Exposure (s) = {camera.shutter_speed / 1000000.}")

        for i in range(numberImages + 1):

            # Capture one image
            image, meta = self.__snap_beta(camera)

            # Write image to file, toss first image
            if i > 0:
                betaImage = LrmImage(image, meta)
                outFileName = f"{append_slash(fullPath)}{filePrefix}_{i}"
                betaImage.save(outFileName)

            print(f"Captured {i} of {numberImages} Duration = {meta['capture_time_s']}")

            # Store for display
            self.lastBetaImage = image

        # Framerate needs to be set to 1 in order to let the camera close at long shutter speeds
        camera.framerate = 1
        camera.close()

    def capture_beta(self, fullPath, filePrefix, numberImages, analogGain, exposureDurationUs, threshold=0):
        """Integrate desired number of frames from camera object and write to disk (save threaded version)
        fullPath: save path
        filePrefix: prefix of saved file
        numberImages: number of frames to capture
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure the LED is off just in case
        self.led.off()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        # Set up camera for beta acquisition
        self.__setup_beta(camera, analogGain, exposureDurationUs)

        # Announce
        # Announce
        print(f"Capturing beta image(s): path = {fullPath} prefix = {filePrefix} # {numberImages}\n"
              f"Gain = {camera.analog_gain} Exposure (s) = {camera.shutter_speed / 1000000.}")


        infoList = []

        for i in range(numberImages + 1):

            # Capture one image

            image, info = self.__snap_beta(camera, threshold)

            # Write image to file, toss first image
            if i > 0:

                infoList.append(info)

                # Sum image
                if i == 1:
                    summed = image
                else:
                    summed = summed + image

                print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(
                    info['capture_time_s']) + " Sum = " + str(image.sum()) + " Total sum = " + str(summed.sum()))

        # Close the camera as soon as we can
        camera.framerate = 1
        camera.close()

        # Write image to file
        betaImage = LrmImage(summed, infoList)
        outFileName = append_slash(fullPath) + filePrefix
        betaImage.save(outFileName)

        # Store for display
        self.lastBetaImage = summed

    def capture_brightfield(self, fullPath, filePrefix, numberImages, analogGain=None, exposureDurationUs=None):
        """Capture one or more brightfield images and write to disk as .png
        fullPath: path to save brightfield images
        filePrefix: prefix of saved images
        numberImages: number of frames to capture        
        analogGain: fixed analog gain setting; automatic if None
        exposureDurationUs: fixed shutter speed; automatic if None
        Returns: nothing
        """

        # Make sure LED is on
        self.led.on()

        # Make destination directory if it doesn't exist
        check_or_make_directory(fullPath)

        # Capture
        camera = picamera.PiCamera()

        self.__setup_brightfield(camera, analogGain, exposureDurationUs)

        # Announce
        print(
            "Capturing brightfield image(s): path = " + fullPath + " prefix = " + filePrefix + " # " + str(numberImages)
            + " Gain = " + str(round(camera.analog_gain, 2))
            + " Exposure (s) = " + str(round(camera.shutter_speed / 1000000., 2)))

        for i in range(numberImages):
            # Capture one image
            image, info = self.__snap_brightfield(camera)
            print("Captured " + str(i) + "/" + str(numberImages) + " Duration = " + str(info['capture_time_s']))

            # Store for display
            self.lastBfImage = image

            # Extract data from camera stream
            outFileFullPath = append_slash(fullPath) + filePrefix + str(i)

            # bfImage = BetaImage(image,info)

            # Close camera as soon as we can
            camera.framerate = 1
            camera.close()

            # Save file as lz4-compressed data
            # disable for now for space savings 5/23/2019 and speed
            # bfImage.save(outFileFullPath)

            # ... and also as .jp2 for quick preview
            bfOutFileFullPath = append_slash(fullPath) + filePrefix + str(i) + ".j2k"

            # Sum along color channel axis and enforce uint16 output
            image = image.astype(np.uint8)
            imageU16 = image.sum(axis=2)
            imageU16 = imageU16.astype(np.uint16)

            # Save to file
            Image.fromarray(imageU16).save(bfOutFileFullPath)

        # Turn LED off
        self.led.off()

    def is_camera_settled(self, camera):
        """Check if the camera autoexposure mode has settled on a gain and shutter setting
        
        returns: most recent gain, shutter speed, and True if camera is settled
        """
        settledPercent = 2.5

        # Get current gain / exposure values
        lastGain = camera.analog_gain
        lastShutterUs = camera.exposure_speed

        # Give the camera some time to settle, based on current exposure value
        settleWaitTime = lastShutterUs / 1000000.
        time.sleep(1 + (settleWaitTime * 5))

        # Get current gain / exposure values
        gain = camera.analog_gain
        shutter = camera.exposure_speed

        # Check if values have change much during wait period
        settled = (is_within_percent(gain, lastGain, settledPercent) and
                   is_within_percent(shutter, lastShutterUs, settledPercent))

        return gain, shutter, settled

    def wait_for_camera_settled(self, camera):
        """ Wait for camera autoexposure mode to settle on analog gain and shutter settings
        returns: settled gain and shutter speed values
        """

        settled = False

        # Loop until the camera has settled
        while not settled:
            lastGain, lastShutterUs, settled = self.is_camera_settled(camera)
            time.sleep(0.1)

        return lastGain, lastShutterUs

    def get_brightfield_exposure(self):
        """Run the camera in autoexposure mode and report gain and shutter speed once stabilized
        
        returns:
            gain: analog gain of camera
            shutterUs: shutter speed in microseconds
        """

        # Turn on LED
        self.led.on()

        # Start up camera 
        camera = picamera.PiCamera()
        camera.iso = 100
        camera.framerate_range = (30, 30)
        camera.resolution = (3280, 2464)
        camera.sensor_mode = 0
        camera.awb_mode = 'auto'
        camera.shutter_speed = 0
        camera.exposure_mode = 'auto'

        # Wait for camera to stabilize and collect stabilized values
        lastGain, lastShutterUs = self.wait_for_camera_settled(camera)

        # Turn off LED
        self.led.off()

        # Close the camera
        camera.close()

        return lastGain, lastShutterUs

    def preview_last(self, hold=False):

        def prepare_image(image, colorMap='gray'):
            displayResolution = (616, 820)
            im = Image.fromarray(image[1140:2140, 882:1882])
            im = im.convert('L')
            # im.thumbnail((820,616))
            im = np.array(im, dtype=np.uint8)
            cmap = cm.get_cmap(colorMap)
            im = cmap(im)
            im = im * 255
            im = im.astype(np.uint8)

            return im

        bfYes = self.lastBfImage is not None
        betaYes = self.lastBetaImage is not None

        if hold == False:
            plt.figure("Preview")
            plt.ion()
            plt.show()
        else:
            plt.figure("Preview")

        if bfYes:
            bf = prepare_image(self.lastBfImage, 'gray')

        if betaYes:
            beta = prepare_image(self.lastBetaImage, 'inferno')

        if bfYes and betaYes:
            merged = bf + beta
            plt.imshow(merged)

        else:
            if bfYes:
                plt.figure()
                plt.imshow(bf)

            else:

                if betaYes:
                    plt.figure()
                    plt.clim(0, 10)
                    plt.imshow(beta)

        if hold == False:
            plt.draw()
            plt.pause(0.01)
        else:
            plt.show()
