import lz4.frame
import numpy as np
import pickle
from fractions import Fraction


class Betafile:

    def __init__(self, fileAndPath=None):
        self.fileAndPath = fileAndPath
        self.imageInfo = {}

        if fileAndPath is not None:
            self.load(fileAndPath)

    def load(self, fileAndPath):
        self.fileAndPath = fileAndPath

        with open(fileAndPath, 'rb') as file:
            imageDict = pickle.load(file)

        imageLZ4 = imageDict['image-lz4']
        decompressed = lz4.frame.decompress(imageLZ4)
        image = np.frombuffer(decompressed, dtype=np.uint16)

        self.imageInfo = imageDict['info']

        resStr = self.imageInfo['resolution']

        self.resolution = list(map(int, resStr.split('x')))

        self.imageNp = image.reshape((self.resolution[1], self.resolution[0]))

    def show(self):
        fig, ax = plt.subplots()
        med = np.median(self.imageNp)
        ax.imshow(self.imageNp, cmap='bone')
        plt.clim(med * 0.5, med * 2)
        plt.show()


