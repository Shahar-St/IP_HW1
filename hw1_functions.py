import cv2
import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
    # print("123456789")
    print("123456789+987654321\n")


def contrastEnhance(im, range):
    # TODO: implement function
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax + 1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1, im2):
    # TODO: implement function
    return d


def meanSqrDist(im1, im2):
    # TODO: implement function - one line
    return d


def sliceMat(im):
    # TODO: implement function
    return Slices


def SLTmap(im1, im2):
    # TODO: implement function
    return mapImage(im1, TM), TM


def mapImage(im, tm):
    # TODO: implement function
    return TMim


def sltNegative(im):
    # TODO: implement function - one line
    return nim


def sltThreshold(im, thresh):
    # TODO: implement function
    return nim
