import numpy as np
import lz4
import lz4.frame
import pickle
import matplotlib
import matplotlib.pyplot as plt
import glob

matplotlib.use('TkAgg')

class LrmFile:
    """Class that writes compressed LRM image and meta data to pickle file"""

    def __init__(self):
        self.frames = []

    def __create_frame(self, image, meta):
        """Returns an image frame which is a dictionary with image and metadata"""
        return      {'image': image,
                     'meta': meta}

    def __compress_frame(self, frame):
        """Returns compressed image frame"""

        frame.image = lz4.frame.compress(frame.image,
                                      return_bytearray=True,
                                      store_size=False)

        return frame


    def __decompress_frame(self, frame):
        """Returns decompressed image frame"""

        # Decompress
        decompressed = lz4.frame.decompress(frame.image)

        # Reshape raw decompressed data into correct np shape
        image = np.frombuffer(decompressed, dtype=np.uint16)

        resolution = list(map(int, frame.meta['resolution'].split('x')))

        frame.image = image.reshape((self.resolution[1], self.resolution[0]))

        return frame


    def add_frame(self, image, meta):
        """Compresses and stores an image and meta data as a frame"""

        frame = self.__create_frame(image, meta)
        compressed = self.__compress_frame(frame)
        self.frames.append(compressed)


    def load(self, file_or_glob):
        """Loads a single file or multiple files if glob is passed"""



    def save(self, file, clear = True, append = True):
        """Saves all frames to a file"""


        # Add .lrm extension to all files
        if not file[-4:] == 'lrm':
            file = file + '.lrm'

        # Pickle it
        with open(file, 'wb') as f:
            pickle.dump(imageDict,f,protocol=pickle.HIGHEST_PROTOCOL)


    def number_frames(self):
        """Returns the number of frames"""
        return len(self.images)



class LrmImage:

    def __init__(self):
        self.frames = []

    @property
    def number_frames(self):
        """Returns the number of frames stores in the file"""

    def load_images(self, file_glob):
        """Loads a stack of LRM images"""

        files = glob.glob(file_glob)

        for f in files:
            frame = self.load(f)

            decomped = self.decompress(frame)

            self.add_frame()


    def __make_frame(self, image, image_type, meta):
        return      {'image': image,
                     'image_type' : image_type,
                     'meta': meta}

    def add_frame(self, image, image_type, meta):

        frame_dict = self.__make_frame(image, image_type, meta)

        self.frames.append(frame_dict)


    def decompress(self, compressed):
        """Decompresses image data and returns dictionary"""

        imageLZ4 = compressed['image-lz4']

        decompressed = lz4.frame.decompress(imageLZ4)

        image = np.frombuffer(decompressed, dtype=np.uint16)

        self.imageInfo = imageDict['info']

        resStr = self.imageInfo['resolution']

        self.resolution = list(map(int, resStr.split('x')))

        self.imageNp = image.reshape((self.resolution[1], self.resolution[0]))


    def compress(self, frame):


    def load(self, file):
        """ Load beta images from file
        """

        if file is not None:
            self.load(file)

        # Load Pickle
        with open(file, 'rb') as file:
            compressed = pickle.load(file)

        self.decompress(compressed)

        with open(filename, 'rb') as file:
            self.imageDict = pickle.load(file)

        # Perform LZ4 decompression of image data
        imageLZ4 = self.imageDict['image']
        decompressed = lz4.frame.decompress(imageLZ4)

        # Convert raw decompressed data into 16 bit numpy
        self.image = np.frombuffer(decompressed, dtype = np.uint16)

        # Grab image information
        self.meta = self.imageDict['info']

        firstImageNfo = self.meta[0]

        # Reshape proper image dimensions based on recorded resolution
        resolution = firstImageNfo['resolution']
        self.image = self.image.reshape((resolution[1],resolution[0]))

        return self

    def save(self, file):
        """ Save images to file
        """

        # Compress numpy image
        imageLZ4 = lz4.frame.compress(self.image, return_bytearray=True, store_size=False)

        # Pack everything into dictionary
        imageDict = {**{'meta': self.info}, **{'image' : imageLZ4 }}s

        # Add .beta extension to all files
        if not file[-4:] == 'beta':
            file = file + '.beta'

        # Pickle it
        with open(file, 'wb') as f:
            pickle.dump(imageDict,f,protocol=pickle.HIGHEST_PROTOCOL)

    def show(self, frame_num = None):
        """ Display the image """

        if frame_num == None:
            frame_num = 1

        fig, ax = plt.subplots()
        ax.imshow(self.frame.image[frame_num])
        plt.clim( self.image.mean() * 0.9 , self.image.mean() * 1.1 )
        plt.show()
















