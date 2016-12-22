#from __future__ import division

from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math


"""
    Below program will compuete entropy of 3 images for each level of intensity i.e; from 0 to 255
    for eg for L = 0 compute entropy of pixels which have intensity upto L Ea and compute the intensity of pixels for L+1 Eb
    now  sum = Ea + Eb and likewise we will have 256 different sums for each intensity level
    From these sums we find the maximum sum value and correxponding entropies Ea and Eb for that sum
    In the end we plot 3 input images along with binaray images obtained by tresholding at level l where the sum is max
"""
def computeL(im):

    """
    below function will compute below parameter
    1)compute the histogram array for each intensity level and probability array which consists of probability of
    pixels of each intensity level for example for intensity level = 0 no of pixels having intensity level = 0 is
    h[0]/total no of pixels  where h is the histogram array

    2)After getting probability array compute entropy for each intensity level using the formula
    E(x) = -sum (h[k]* log(h[k])) where k is from 0 to 255

    3)from these sums array get the sum value which is maximum and its corresponding L value
     return that L value
    :param im:
    :return:
    """


    I = np.array(im)
    L = 0
    total_count = 0
    hist = [0]*256
    prob = [0]*256

        #below block will compuete histogram array and probability array for each intensity level

    rows = I.shape[0]
    #print rows
    cols = I.shape[1]
    #print cols
    for m in range(0, rows):
        for n in range(0, cols):
            temp = I[m][n]
            hist[I[m][n]] = hist[I[m][n]] +1


    max_value = 0
    for s in range(0,256):
        prob[s] = float(hist[s])/(I.shape[0]*I.shape[1])

    total = []
    index = -1
    for i in range(0,256):
        sump = 0
        prob_bg = []
        prob_fg = []
        for n in range(0,i+1):
            sump = sump + prob[n]
        value1 = 0
        for x in range(0,i+1):

            if (sump == 0):
                value = 0
            else:
                value1 = float(prob[x])/sump
            prob_bg.append(value1)
        value2 = 0
        for y in range(i+1,256):

            if (sump == 1):
                value2 = 0
            else:
                value2 = float(prob[y])/(1-sump)
            prob_fg.append(value2)

        sumea = 0
        sumeb = 0


        for x in range(0,len(prob_bg)):

            if(prob_bg[x] == 0):

                sumea += 0

            else:

                sumea += prob_bg[x]*np.log2(prob_bg[x])

        for y in range(0,len(prob_fg)):

            if(prob_fg[y] == 0):

                sumeb += 0

            else:

                sumeb += prob_fg[y]*np.log2(prob_fg[y])

        #total.append(-(sumea) -(sumeb))

        if(max_value<-(sumea) -(sumeb)):
            max_value = -(sumea) -(sumeb)
            index = i


    #print total
    #print max(total)
    #print max_value
    #print index
    list_1 = []
    #treshold the image based on the index value ,pixel values which are  greater then threshold level are white pixels and others are black pixels
    I1 = 1*(I>index)
    list_1.append(I1)
    list_1.append(index)
    return list_1

imagelist = []
imagelist.append(Image.open('cat.jpg').convert("L"))
imagelist.append(Image.open('mountain.jpg'))
imagelist.append(Image.open('waterfall.jpg'))
#imagelist.append(Image.open('eee.jpg'))

intensitylist = []
bin_imagelist = []
# imy = np.ones(I1[0].shape, np.float32)

for i in range(0,3):
    bin_img, intensity = computeL(imagelist[i])
    intensitylist.append(intensity)
    bin_imagelist.append(bin_img)
    #bin_imagelist.append(gen_bin_image(intensitylist[i],np.array(imagelist[0])))
    print "Intensity value for image ", i, "is :", intensitylist[i]
    print "\n"


#plot each input image and binary image in single window

for i in range(0,3):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([.3, .55, .35, .35])
    image1 = (imagelist[i]).convert("L")
    arr = np.array(image1)
    # plt.imshow(arr, cmap = cm.Greys_r)
    ax.imshow(arr, cmap = cm.Greys_r)
    plt.gray()
    ax.autoscale(False)
    ax2 = fig.add_axes([.3,  .05, .35, .35], sharex=ax, sharey=ax)
    plt.gray()
    ax2.imshow(bin_imagelist[i])
    ax2.autoscale(False)

plt.show()


