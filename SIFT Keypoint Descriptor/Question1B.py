from __future__ import division
import Queue
from imaplib import _Authenticator

from sys import  argv
"""
    Below program will implement Harris corner Detection Algorithm
    It smoothes the image using guassian filter
    computes first derivative Ixx ,Iyy and Ixy of the smoothed image
    computes Hessian matrix for each pixel of the image
    computes determinant and trace of each pixel of the image
    corner-ness value is measured based on det - (alpha*trace)
    If it is greater then threshold  of 0.1 and minimum distance of 10 then those points (x,y) locations
    are retained in the list.
    All corners location of the list are  plotted on the image
"""

from PIL import Image
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from  scipy.ndimage import filters
import math





def harriscorner(image_name,counter):
    min_dist = 20
    threshold=0.1
    i2 = Image.open(image_name).convert('I')
    im = np.array(i2)
    sigma = 1

    #below step will perform guassian blurring with sigma on image im
    Ig = np.zeros(im.shape)
    Ig = filters.gaussian_filter(im,sigma)

    #below step will display the image

    #below step will compute first derivative in x direction  on smoothed image
    Ix = np.zeros(Ig.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), Ix)



    #below step will compute first derivative in y direction  on smoothed image
    Iy = np.zeros(Ig.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), Iy)

    # below steps will compute components of harris matrix
    Ixx = Ix*Ix
    filters.gaussian_filter(Ixx,sigma)

    Ixy = filters.gaussian_filter(Ix*Iy,sigma)

    Iyy = filters.gaussian_filter(Iy*Iy,sigma)

    heissian = []

    for i in range(0,im.shape[0]):
        each_row = []
        for j in range(0,im.shape[1]):
            h_eachpixel = np.zeros((2,2))
            h_eachpixel[0][0] = Ixx[i][j]
            h_eachpixel[0][1] = Ixy[i][j]
            h_eachpixel[1][0] = Ixy[i][j]
            h_eachpixel[1][1] = Iyy[i][j]
            each_row.append(h_eachpixel)
        heissian.append(each_row)

    #computes determinant and trace of the hessian matrix
    det = Ixx*Iyy - Ixy**2
    tr = Ixx + Iyy
    harrisim = np.array(det) - np.array(tr)*alpha


    corner_threshold = np.max(harrisim) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    locations = np.array(harrisim_t.nonzero()).T

    candidate_values = [harrisim[x[0],x[1]] for x in locations]

    # sort candidates and get the index of maximum value
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)

    #from -minimum distanc to +minimumdistance allocate 1 to mask array
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    required_coords = []
    m = 0
    for i in index:
        if m==0:
            print(i)
            m = 1
        if allowed_locations[locations[i,0],locations[i,1]] == 1:
            required_coords.append(locations[i])
            # if a initial harris location is found then mask neighbouring location with in distance of minimum distance
            #zero so that next probable harris corner location will be found at a distance of 10
            allowed_locations[(locations[i,0]-min_dist):(locations[i,0]+min_dist),
            (locations[i][1]-min_dist):(locations[i][1]+min_dist)] = 0

    #plot the harris points
    plt.figure(counter)
    plt.gray()
    plt.imshow(i2)
    plt.plot([p[1] for p in required_coords],[p[0] for p in required_coords],'*')

    #return required_coords



#Loads the input image
image_name = ['input1.png','input2.png','input3.png']

alpha = 0.04
counter = 1
i = 0
while i<3:
    harriscorner(image_name[i],i+1)
    i+=1
plt.show()
print "alpha = 0.04 and minimum distance of 10 we were able to get approximately good harris corners as alpha increases the threshold value" \
             "cornerness decreases and we will get more number of corner which are not real corners"


