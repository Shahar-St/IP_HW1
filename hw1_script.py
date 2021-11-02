from hw1_functions import *
from cv2 import cv2

if __name__ == "__main__":

    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    maxRangeList = [0, 255]
    enhanced_img, a, b = contrastEnhance(darkimg_gray, maxRangeList)

    # display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # print a,b
    print("a = {}, b = {}\n".format(a, b))

    # display mapping
    hist, _ = np.histogram(darkimg_gray, bins=256, range=(0, 255))
    hist = hist.astype(float)
    darkimg_gray_RangeList = [np.min(darkimg), np.max(darkimg)]
    showMapping(darkimg_gray_RangeList, a, b)  # add parameters

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(enhanced_img, maxRangeList)
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    d = minkowski2Dist(enhanced_img, enhanced2_img)
    print("Minkowski dist between enhanced image and its enhancement (expected: 0)")
    print("dist = {}\n".format(d))
    mse = meanSqrDist(enhanced_img, enhanced2_img)
    print("MSE of enhanced image and its enhancement (expected: 0)")
    print("mse = {}\n".format(mse))

    print("c ------------------------------------\n")
    mdist = minkowski2Dist(darkimg_gray, darkimg_gray)
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))

    # implement the loop that calculates minkowski distance as function of increasing contrast
    darkimg_gray_minRange = darkimg_gray_RangeList[0]
    step = (darkimg_gray_RangeList[1] - darkimg_gray_RangeList[0]) / 20

    contrasts, dists = [], []
    for k in range(1, 21):
        contrast = darkimg_gray_minRange + round(k * step)
        contrasts.append(contrast)
        img_enhanced, _, _ = contrastEnhance(darkimg_gray, [darkimg_gray_minRange, contrast])
        dists.append(minkowski2Dist(darkimg_gray, img_enhanced))

    plt.figure()
    plt.plot(contrasts, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")
    # computationally prove that sliceMat(im) * [0:255] == im
    check_im = np.matmul(sliceMat(darkimg_gray), np.transpose(np.arange(256)))
    check_im = check_im.reshape(np.shape(darkimg_gray)[0], np.shape(darkimg_gray)[1])
    d = minkowski2Dist(darkimg_gray, check_im)
    print("Minkowski distance: {}".format(d))
    print(f"sliceMat(im) * [0:255] == im ? {np.all(check_im == darkimg_gray)}")

    print("e ------------------------------------\n")

    max_contrast_darkimg_gray, _, _ = contrastEnhance(darkimg_gray, maxRangeList)  # computationally compare
    TMim, TM = SLTmap(darkimg_gray, max_contrast_darkimg_gray)
    d = minkowski2Dist(check_im, max_contrast_darkimg_gray)
    print("sum of diff between image and slices*[0..255] = {}".format(d))

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.show()
    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")

    print("g ------------------------------------\n")
    thresh = 120  # play with it to see changes
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold(lena_gray, thresh)  # add parameters

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")

    print("h ------------------------------------\n")
    im1 = lena_gray
    im2 = darkimg
    SLTim, _ = SLTmap(im1, im2)

    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im1)
    plt.title("original image")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")

    d1 = meanSqrDist(im1, im2)  # mean sqr dist between im1 and im2
    d2 = meanSqrDist(SLTim, im2)  # mean sqr dist between mapped image and im2
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))
    if d1 > d2:
        print("smaller!")
    else:
        print("not smaller!")

    print("i ------------------------------------\n")
    # prove comutationally
    SLTmap(im1, im2)
    SLTmap(im2, im2)
    d = 7 # TODO:
    print(" {}".format(d))

    plt.show()