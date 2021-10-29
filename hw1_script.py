from hw1_functions import *


if __name__ == "__main__":
    # feel free to add/remove/edit lines

    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()

    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance()#add parameters

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

    #display mapping
    showMapping()#add parameters

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance()#add parameters
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))
    
    # TODO: display the difference between the two image (Do not simply display both images)




    print("c ------------------------------------\n")
    mdist = minkowski2Dist()#add parameters
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))

    # TODO:
    # implement the loop that calculates minkowski distance as function of increasing contrast

    contrast = # TODO
    dists = # TODO


    plt.figure()
    plt.plot(contrast, dists)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")

    print("d ------------------------------------\n")
    

    d = # computationally prove that sliceMat(im) * [0:255] == im
    print("".format(d))



    print("e ------------------------------------\n")
    
    d = # computationally compare
    print("sum of diff between image and slices*[0..255] = {}".format(d))

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(TMim, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")



    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")



    print("g ------------------------------------\n")
    thresh = 120 # play with it to see changes
    lena = cv2.imread(r"Images\\RealLena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    thresh_im = sltThreshold()#add parameters

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")



    print("h ------------------------------------\n")
    im1 = lena_gray
    im2 = darkimage
    SLTim = #TODO

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



    d1 = # mean sqr dist between im1 and im2
    d2 = # mean sqr dist between mapped image and im2
    print("mean sqr dist between im1 and im2 = {}\n".format(d1))
    print("mean sqr dist between mapped image and im2 = {}\n".format(d2))

    print("i ------------------------------------\n")
    # prove comutationally
    d = # TODO:
    print(" {}".format(d))


    plt.show()