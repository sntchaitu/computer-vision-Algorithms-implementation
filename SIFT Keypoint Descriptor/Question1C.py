from __future__ import division
import Queue
from imaplib import _Authenticator

from sys import  argv
"""
    Below program will implement Harris corner Detection Algorithm using minium eigen value method
    computes first derivative Ixx ,Iyy and Ixy of the smoothed image
    computes Hessian matrix for each pixel of the image
    computes eigen values for hessian matrix of each pixel
    corner-ness value is measured based on  lambda1*lambda2 - alpha(lambda1+lambda2)
      If it is greater then threshold  of 0.1 and minimum distance of 10 then those points (x,y) locations
    are retained in the list.
    All corners location of the list are  plotted on the image
"""

from PIL import Image
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from  scipy.ndimage import filters
from scipy import misc
from scipy import  signal
import math
from numpy.linalg import inv
import copy as cp
from numpy.linalg import inv
from scipy import interpolate
from scipy import ndimage
import cv2
import time



def harriscorner(image_name,counter):
    #threshold value for the cornerness measures as the hessian is computed for 3*3 window the value of this threshold

    #is more in order to filter less corners
    threshold=5000000
    lambda_treshold = 199000
    i2 = Image.open(image_name).convert('I')
    i1 = Image.open(image_name)
    im = np.array(i2)
    sigma = 1
    list1 = []

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


    epsi = 0.001
    #for each  3*3 window compute the heissian matrix
    for i in range(1,im.shape[0]-1):
        for j in range(1,im.shape[1]-1):
            hess = np.zeros((2,2))

            ixx_temp  = Ixx[i-1:i+1+1, j-1:j+1+1]
            iyy_temp  = Iyy[i-1:i+1+1, j-1:j+1+1]
            ixy_temp = Ixy[i-1:i+1+1, j-1:j+1+1]
            hess[0][0] = epsi+np.sum(ixx_temp)
            hess[0][1] = epsi+np.sum(ixy_temp)
            hess[1][0] = epsi+np.sum(ixy_temp)
            hess[1][1] = epsi+np.sum(iyy_temp)
            eigen_value = np.linalg.eigvals(hess)
            l1 = eigen_value[0]
            l2 = eigen_value[1]
            if  (l1*l2 - (0.04)*(l1+l2))>threshold :
                list1.append([i,j])


    # store allowed point locations in array
    allowed_locations = np.zeros(im.shape)
    min_dist = 21
    #from -minimum distanc to +minimumdistance allocate 1 to mask array
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    # select  points at equal distance  taking min_distance into account
    required_coords = []
    m = 0
    for i in list1:
        if m==0:
            print(i)
            m = 1
        if allowed_locations[i[0],i[1]] == 1:
            required_coords.append(i)
            # if a initial harris location is found then mask neighbouring location with in distance of minimum distance
            #zero so that next probable harris corner location will be found at a distance of min_dist
            allowed_locations[(i[0]-min_dist):(i[0]+min_dist),
            (i[1]-min_dist):(i[1]+min_dist)] = 0


    plt.figure(counter)
    plt.imshow(i1)
    plt.plot([p[1] for p in required_coords],[p[0] for p in required_coords],'go')

    #plt.show()



#Loads the input image
image_name = ['input1.png','input2.png','input3.png']

counter = 1
i = 0
while i<3:
    st_time = time.time()
    harriscorner(image_name[i],i+1)
    i+=1
    print" execution time for image ",i,"is ",time.time()-st_time
plt.show()
print "for  each image we get approximately same corners as of previous question but the amount of computation time is much larger compared" \
      "to previous method "


