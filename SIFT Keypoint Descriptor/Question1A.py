from __future__ import division
import Queue
from imaplib import _Authenticator

from sys import  argv
"""
    Below program will implement Harris corner Detection Algorithm using minium eigen value method
    computes first derivative Ixx ,Iyy and Ixy of the smoothed image
    computes Hessian matrix for each pixel of the image
    computes eigen values for hessian matrix of each pixel
    If it is greater then threshold  of 199000  then those points (x,y) locations
    are retained in the list.
    All corners location of the list are  plotted on the image
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from  scipy.ndimage import filters



def harriscorner(image_name,counter):
    threshold=0.1
    lambda_treshold = 199000
    i2 = Image.open(image_name).convert('I')
    i1 = Image.open(image_name)
    im = np.array(i2)
    sigma = 1
    list1 = []

    Ix = np.zeros(im.shape)
    Iy = np.zeros(im.shape)
    Ixx = np.zeros(im.shape)
    Iyy = np.zeros(im.shape)
    Ixy = np.zeros(im.shape)

    #compute Ix,Iy,Ixx,Iyy,Ixy using prewitt filter
    Ix  = filters.prewitt(im,0)
    Iy  = filters.prewitt(im)
    Ixx = filters.prewitt(Ix,0)
    Iyy = filters.prewitt(Iy)
    Ixx, Ixy = np.gradient(Ix)


    epsi = 0.001
    #for a 3*3 window compute the heissian matrix
    for i in range(1,im.shape[0]-1):
        for j in range(1,im.shape[1]-1):
            hess = np.zeros((2,2))

            ix_temp = Ix[i-1:i+1+1, j-1:j+1+1]
            iy_temp = Iy[i-1:i+1+1, j-1:j+1+1]

            hess[0][0] = epsi+np.sum(np.multiply(ix_temp,ix_temp))
            hess[0][1] = epsi+np.sum(np.multiply(ix_temp,iy_temp))
            hess[1][0] = epsi+np.sum(np.multiply(ix_temp,iy_temp))
            hess[1][1] = epsi+np.sum(np.multiply(iy_temp,iy_temp))
            eigen_value = np.linalg.eigvals(hess)
            if  eigen_value[1] > lambda_treshold and  eigen_value[0] > lambda_treshold :
                list1.append([i,j])

    plt.figure(counter)
    plt.imshow(i1)
    plt.plot([p[1] for p in list1],[p[0] for p in list1],'*')
    #plt.show()




#Loads the input image
image_name = ['input1.png','input2.png','input3.png']

counter = 1
i = 0
while i<3:
    harriscorner(image_name[i],i+1)
    i+=1
plt.show()
print "for  minimum eigen value thresold of  we were able to get approximately good harris corners "


