from __future__ import division
import Queue
from imaplib import _Authenticator
from sys import  argv
from PIL import Image
from PIL import ImageChops as ic
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


#below program will compute Optical flow in pair of image pairs as below steps using Guassian

#1   first we compuete 3 levels of pyramid for image A and B

#2   start at  least resolution level compuete u and v flow vectors by perform Lucas Kanade step for 3 iterations
#3   upsample the both flow vectors and increase the magnitude by 2
#4   imwarp the image of greater resolution level with the Upsampled U 5 V values and now send this pyramid level A and B image
#5 Now apply lucas kanade step on this warped image B with respect to A of same pyramid level




#Details of Lucas kanade step

#first we compute harris corners of one image uisnsg goodfeatures to track .It will give the list of harris corners
#we compuete gradient of image and dt using temporal difference between 2 images

#next we run a 5*5 window around each harris corner and compute  sum of fx*fx   fy*fy  and fxfy ,fxft and fyft values in each of the window and
#and we get A = [[fx*fx;fxfy][fy*fx;fyfy]] and B = [fxft fyft]transpose
#and perform (inv(A))*(-B) on each pixel and we get [u v]transpose

#next we plot [u v] on the both image pairs using plt.arrow funtion

# params selection  for  corner detection like no of corners , threshold value and mindistance between corners and size of filter block that is passed in computing corners







feature_params = dict( maxCorners = 3000,qualityLevel = 0.001,minDistance = 4,blockSize = 5)

def gpyramid (Atemp,levels):

#below function will guassian pyramid for 3 levels

    plist = []
    Acopy = np.zeros(Atemp.shape)
    Acopy[:] = Atemp
    plist.append(Acopy)
    for z in range(0,levels-1):
        h = int(plist[len(plist)-1].shape[0]/2)
        w = int(plist[len(plist)-1].shape[1]/2)
        r_image = np.zeros((h,w))
        cv2.pyrDown(plist[len(plist)-1],r_image)
        plist.append(r_image)


    return plist


def compueteHarrisforeachLevel(Atemp1,levels,halfwindow):


    """
        below function will compuete harris corners of the image and reduce the corners co-ordinates by half for respective lower resolution levels
    :param Atemp1:
    :param levels:
    :return:clist which is the list of corners
    """

    # Apad[:] = np.pad(A,halfwindow,mode='reflect')
    # Bpad[:] = np.pad(B,halfwindow,mode='reflect')
    clist = []
    cornersList = []
    Acopy1 = np.zeros((Atemp1.shape[0]+2*halfwindow,Atemp1.shape[1]+2*halfwindow),dtype = np.float32)
    #Acopy1 = np.zeros(Atemp1.shape,dtype=np.float32)
    Acopy1[:] = np.pad(Atemp1,halfwindow,mode='reflect')
    p0 = cv2.goodFeaturesToTrack(Acopy1, mask = None, **feature_params)
    pt = np.zeros(p0.shape)
    pt = cp.deepcopy(p0)
    clist.append(pt)
    #reducce each corner x and y co-ordinates by 2
    for z1 in range(0,levels-1):
        ptemp = np.zeros(p0.shape)
        for i in range (0 , p0.shape[0]):
            ptemp[i][0][1] = p0[i][0][1]/2
            ptemp[i][0][0] = p0[i][0][0]/2
        p0 = cp.deepcopy(ptemp)
        clist.append(ptemp)

            #reduce the image size

    return clist

def calcLK(A1,B1,halfwindow,alpha,levels):

    ###
        #this function will compuete lucas kanade for 3 pyramid velvls and for each pyramid vele 3 iterations
    ###

    prrA = []
    pyrB = []
   #compute pyramid for image A and image B
    pyrA = gpyramid(A1,levels)
    pyrB = gpyramid(B1,levels)
    #compuete harris corens at the high resolution image and reduce the size by 2 for otherlevels and store them in a list
    hPyrA  = compueteHarrisforeachLevel(A1,levels,halfwindow)
    hPyrB  = compueteHarrisforeachLevel(B1,levels,halfwindow)


    #for each pyramid level iteration  compuete Lk step
    for pyriter in range(2,-1,-1):

        if pyriter == 2:

            #u and v image for to store vectors
            u = np.zeros((pyrB[pyriter].shape[0],pyrB[pyriter].shape[1]),dtype=np.float32)
            v = np.zeros((pyrB[pyriter].shape[0],pyrB[pyriter].shape[1]),dtype=np.float32)

        #pad the image with haof window
        Apad = np.zeros((pyrA[pyriter].shape[0]+2*halfwindow,pyrA[pyriter].shape[1]+2*halfwindow))
        Apad[:] = np.pad(pyrA[pyriter],halfwindow,mode='reflect')
        B = np.zeros(pyrB[pyriter].shape)
        B[:] = pyrB[pyriter]

        B_original = np.zeros(B.shape)
        B_original[:] = B
        #pad the image A with half of window size for border pixels
        Bpad = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))
        # LK step will be computed for 3 iterations

        for iter in range(0,3):
            Ix = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))
            Iy = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))
            It = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))



            #print "It shape is",It.shape
            #if the iteration is the second remap the image
            if iter >0:
                B[:] = cv2.remap(B_original,u,v,cv2.INTER_LINEAR)

            #pad the image with im.pad for border pixels
            Bpad[:] = np.pad(B,halfwindow,mode='reflect')

            kernelx = np.array([[-1.0,1.0],[-1.0,1.0]] ,np.float32) # kernel should be floating point type.
            kernely = np.array([[-1.0,-1.0],[1.0,1.0]])
            k1 = np.array([[1.0,1.0],[1.0,1.0]])
            k2 = np.array([[-1.0,-1.0],[-1.0,-1.0]])

            Ix,Iy = np.gradient(Bpad)

            # Dx1,Dy1 = np.gradient(Bpad)
            # Dx2,Dy2 = np.gradient(Apad)
            #
            # Ix = Dx1+Dx2
            #
            # Iy = Dy1+Dy2

            # Ix = signal.convolve2d(Bpad,kernelx,mode='valid')
            # Iy = signal.convolve2d(Bpad,kernely,mode='valid')


            #convolve the image with K1 and k2
            temp1 = signal.convolve2d(Bpad,k1,mode='same')
            temp2 = signal.convolve2d(Apad,k2,mode='same')

            # print "temp1 shape is",temp1.shape
            # print "temp2 shape is",temp2.shape
            # It[:] = temp1+temp2


            #Get the temporal different It betwenn 2 images
            It[:] =  np.subtract(Bpad,Apad)
            It[It<0] = 0

            us = np.zeros((Apad.shape[0],Apad.shape[1]))
            vs = np.zeros((Apad.shape[0],Apad.shape[1]))



            # for i in range(halfwindow,Apad.shape[0]-halfwindow):
            #     for j  in range(halfwindow,Bpad.shape[1]-halfwindow):

            #for each harris corner point we compuete flow around 3*3 matrix
            len1 = hPyrB[pyriter].shape[0]
            for pt in range(0,len1):

                i = hPyrB[pyriter][pt][0][1]
                j = hPyrB[pyriter][pt][0][0]

                A1 = np.zeros((2,2))
                B1 = np.zeros((2,))

                ix = Ix[i-halfwindow:i+halfwindow+1, j-halfwindow:j+halfwindow+1]
                iy = Iy[i-halfwindow:i+halfwindow+1, j-halfwindow:j+halfwindow+1]
                it = It[i-halfwindow:i+halfwindow+1, j-halfwindow:j+halfwindow+1]

                A1[0][0] = alpha+np.sum(np.multiply(ix,ix))
                A1[0][1] = alpha+np.sum(np.multiply(ix,iy))
                A1[1][0] = alpha+np.sum(np.multiply(ix,iy))
                A1[1][1] = alpha+np.sum(np.multiply(iy,iy))
                B1[0] = np.sum(np.multiply(ix,it))
                B1[1] = np.sum(np.multiply(iy,it))

                C = np.dot(np.linalg.pinv(A1),-B1)

                us[i][j] = C[0]
                vs[i][j] = C[1]


                #flow vectors will be summed up and added for all the iterations
            u[:] = u+us[halfwindow:Apad.shape[0]-halfwindow,halfwindow:Apad.shape[1]-halfwindow]
            v[:] = v+vs[halfwindow:Apad.shape[0]-halfwindow,halfwindow:Apad.shape[1]-halfwindow]

            # u = u[u>-halfwindow]
            # u = u[u<halfwindow]


        #it the image pyramid is not the finest resolution  we  upsample the flow vectors
        if pyriter !=0:
            u = 2 * ndimage.zoom(u,2,order=1)
            v = 2 * ndimage.zoom(v,2,order=1)


    #append the and U and V to UV
    uv = []
    uv.append(u)
    uv.append(v)
    return uv

#list of images
imList1 = ['basketball1.png','teddy1.png','grove1.png']
imList2 = ['basketball2.png','teddy2.png','grove2.png']


#for each image  compute entire LK step
for im_iter in range(0,3):

    I1 = Image.open(imList1[im_iter]).convert('L')
    I2 = Image.open(imList2[im_iter]).convert('L')




# image1 = 'grove1.png'
# image2 = 'grove2.png'
    windowsize = 5
    halfwindow  = int(windowsize/2)



# I1 = Image.open(image1).convert('L')
# I2 = Image.open(image2).convert('L')



    A = np.array(I1)
    B = np.array(I2)
    check = np.zeros(B.size)

    levels = 3


    res = calcLK(A,B,halfwindow,0.001,levels)

    u = np.zeros((A.shape[0],A.shape[1]),dtype=np.float32)
    v = np.zeros((A.shape[0],A.shape[1]),dtype=np.float32)
    u[:] = res[0]
    v[:] = res[1]

    print "u shape is",u.shape
    print "v shape is",v.shape
    print "B shape is",B.shape
# check[:] = cv2.remap(B,u,v,cv2.INTER_LINEAR)
#
# plt.figure(0)
# plt.imshow(Image.fromarray(check))



###########################plot the image using stream plot vectos will display the optical flow#####################################################
    spacing  = 1
    print "a shape is",A.shape
    print "u spahe is",u.shape
    print "v spahe is",v.shape

    Y, X = np.mgrid[0:A.shape[0], 0:A.shape[1]]

    u1 = u[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    v1 = v[0:v.shape[0]:spacing,0:v.shape[1]:spacing]

    x1 = X[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    y1 = Y[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    plt.figure(2*im_iter)
    plt.imshow(I1)
    plt.gray()

    speed = np.sqrt(u1*u1 + v1*v1)


    lw = 1*speed/speed.max()

    print "x1 shape is",x1.shape
    print "u1 shape is",u1.shape

    plt.streamplot(x1, y1, u1, v1, color=u1, linewidth=1, cmap=plt.cm.autumn)
    plt.colorbar()

    # plt.figure(im_iter*2)
    # plt.imshow(I2)
    # plt.gray()


    # for i in range(0,u.shape[0]):
    #     for j in range(0,u.shape[1]):
    #         if(u[i][j]!=0.0 or v[i][j]!=0.0):
    #             plt.arrow(j,i,u[i][j],v[i][j],color = 'red',head_width = 6,head_length = 7)
    #
    #     # plotting  optical flow on image2
    # plt.figure(im_iter*2+1)
    # plt.imshow(I1)
    # plt.gray()
    #
    # for i in range(0,u.shape[0]):
    #     for j in range(0,u.shape[1]):
    #         if(u[i][j]!=0.0 or v[i][j]!=0.0):
    #             plt.arrow(j,i,u[i][j],v[i][j],color = 'yellow',head_width = 6,head_length = 7)


plt.show()

