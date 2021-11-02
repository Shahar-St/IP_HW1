import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
    print("305237257+312162027\n")


def contrastEnhance(im, im_range):
    min_im_val = np.min(im)
    max_im_val = np.max(im)

    min_target_val = im_range[0]
    max_target_val = im_range[1]

    a = (max_target_val - min_target_val) / (max_im_val - min_im_val)
    b = min_target_val - (min_im_val * a)

    nim = np.copy(im)
    for (x, y), value in np.ndenumerate(im):
        nim[x][y] = a * value + b
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
    # if not isSameSize(im1, im2):
    #     return -1
    hist1, _ = np.histogram(im1, bins=256, range=(0, 255), density=True)
    hist2, _ = np.histogram(im2, bins=256, range=(0, 255), density=True)

    hist1 = hist1.astype(float)
    hist2 = hist2.astype(float)

    p = 2
    d = np.power(np.sum(np.power(np.abs(np.subtract(hist1, hist2)), p)), 1 / p)

    return d


def meanSqrDist(im1, im2):
    if not isSameSize(im1, im2):
        return -1
    d = np.mean(np.power(np.subtract(im1.astype(float), im2.astype(float)), 2))
    return d


def sliceMat(im):
    slices = []
    im = np.squeeze(im.reshape((im.size, 1)))
    for color in range(256):
        mask = (im == color)
        slices.append(np.copy(mask))
    return np.transpose(slices)


def SLTmap(im1, im2):
    if not isSameSize(im1, im2):
        return -1

    im_slm = sliceMat(im1)
    shaped_im2 = np.squeeze(im2.reshape((im2.size, 1)))

    TM = np.zeros(256)
    color = 0
    for col_color in im_slm.T:
        target_col_sum = np.matmul(col_color, shaped_im2)
        if target_col_sum != 0:
            target_col_mean = (target_col_sum / col_color.sum()).astype(np.int)
            TM[color] = target_col_mean
        color += 1

    return mapImage(im1, TM), TM


def mapImage(im, tm):
    slices = sliceMat(im)
    return np.reshape(np.matmul(slices, tm), (np.shape(im)[0], np.shape(im)[1]))


def sltNegative(im):
    return mapImage(im, np.arange(256)[::-1])


def sltThreshold(im, thresh):
    TM = np.arange(256)
    TM[:thresh] = 0
    TM[thresh:] = 255
    return mapImage(im, np.arange(255)[:thresh])


def isSameSize(im1, im2):
    if np.shape(im1)[0] != np.shape(im2)[0] or np.shape(im1)[1] != np.shape(im2)[1]:
        print("images need to be the same size!")
        return 0

    return 1
