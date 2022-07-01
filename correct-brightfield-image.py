import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math
#from betaprocess import Betafile
from betaimage import BetaImage
from multiprocessing.pool import ThreadPool
import skimage
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
import lz4
import os
import seaborn as sns


def load_uniformity_image():
    # path to unformity measurement
    pathUniformity = "/Users/jjj/Dropbox/Manuscripts/Direct beta detection/Experiments/uniformity-images-6-13-2019/bf-uniformity-lcd-6-13-2019/bf-1.j2k"
    uniformity = Image.open(pathUniformity)

    uni = np.array(uniformity)

    uni = uni / uni.max()

    return uni

# proces image files
path="/Users/jjj/Dropbox/Manuscripts/Direct beta detection/Experiments/fdg-cells-coverslip-5-16-2019/fdg-cells-coverglass-5-16-2019-combined/bf/"
prefix = ''

globDir = path + prefix + '*'
files = glob.glob(globDir)

bfTiffOut = "../bf-tiff/"

correctionImage = load_uniformity_image()

for i,f in enumerate(files):
    fileName = os.path.basename(f)[:-4]
    filePath = os.path.dirname(f)

    print(f)

    j2kImage = np.array(Image.open(f))

    j2kImage = j2kImage.sum(2).astype(dtype=np.float32)

    im = ( j2kImage * correctionImage)

    # write file out

    outFile = filePath + '/' + bfTiffOut + fileName + '.tiff'

    Image.fromarray(im).save(outFile)


