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
import cv2


#below program will compute Optical flow in pair of image pairs as below steps

#first we compute harris corners of one image uisnsg goodfeatures to track .It will give the list of harris corners
#we compuete gradient of image and dt using temporal difference between 2 images

#next we run a 5*5 window around each harris corner and compute  sum of fx*fx   fy*fy  and fxfy ,fxft and fyft values in each of the window and
#and we get A = [[fx*fx;fxfy][fy*fx;fyfy]] and B = [fxft fyft]transpose
#and perform (inv(A))*(-B) on each pixel and we get [u v]transpose

#next we plot [u v] on the both image pairs using plt.arrow funtion

# params selection  for  corner detection like no of corners , threshold value and mindistance between corners and size of filter block that is passed in computing corners

feature_params = dict( maxCorners = 500,qualityLevel = 0.02,minDistance = 7,blockSize = 7)




def calcLK(Image2,A,B,halfwindow,alpha):
    """

    below funtion will compuete optical flow of the image and return the flow vector
    :param Image2:
    :param A:
    :param B:
    :param halfwindow:
    :param alpha:
    :return: u,v flow vector as list
    """



    B_original = np.zeros(B.shape)
    B_original[:] = B

    #u,v matrixes for storing flow
    u = np.zeros((A.shape[0],A.shape[1]),dtype=np.float32)
    v = np.zeros((A.shape[0],A.shape[1]),dtype=np.float32)


    #Lucas Kanade step will be perfomed for 5 iterations and flow of ach iteration is added to the original flow
    for iter in range(0,5):


        Apad = np.zeros((A.shape[0]+2*halfwindow,A.shape[1]+2*halfwindow),dtype = np.float32)
        Bpad = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow),dtype = np.float32)



        Ix = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))
        Iy = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))
        It = np.zeros((B.shape[0]+2*halfwindow,B.shape[1]+2*halfwindow))



        print "It shape is",It.shape

        #if for second iteration we adjust the image based on intial flow vectors
        if iter >0:
            #y1,x1 = np.meshgrid[0:B_original.shape[0],0:B_original.shape[1]]

            #B[:] = interpolate.interp2d(B_original,x1+u, y1+v, 'cubic');
            #B[:] = cv2.resize(B_original,(u,v), interpolation = cv2.INTER_CUBIC)
            B[:] = cv2.remap(B_original,u,v,cv2.INTER_CUBIC)

        #both images are padded
        Apad[:] = np.pad(A,halfwindow,mode='reflect')
        Bpad[:] = np.pad(B,halfwindow,mode='reflect')


        # kernelx = np.array([[-1.0,1.0],[-1.0,1.0]] ,np.float32) # kernel should be floating point type.
        # kernely = np.array([[-1.0,-1.0],[1.0,1.0]])
        k1 = np.array([[1.0,1.0],[1.0,1.0]])
        k2 = np.array([[-1.0,-1.0],[-1.0,-1.0]])

        #Ix,Iy = np.gradient(Bpad)

        #computes the gradient of both images and add it to to get Ix and Iy
        Dx1,Dy1 = np.gradient(Bpad)
        Dx2,Dy2 = np.gradient(Apad)

        Ix = Dx1+Dx2

        Iy = Dy1+Dy2

        # Ix = signal.convolve2d(Bpad,kernelx,mode='valid')
        # Iy = signal.convolve2d(Bpad,kernely,mode='valid')
        temp1 = signal.convolve2d(Bpad,k1,mode='same')
        temp2 = signal.convolve2d(Apad,k2,mode='same')

        # print "temp1 shape is",temp1.shape
        # print "temp2 shape is",temp2.shape
        # It[:] = temp1+temp2

        #compuetes temoral diff between 2 images
        It[:] =  np.subtract(Bpad,Apad)
        It[It<0] = 0


        us = np.zeros((Apad.shape[0],Apad.shape[1]))
        vs = np.zeros((Apad.shape[0],Apad.shape[1]))

        # get harris corner points using goodfeaturestotack
        #[a b] = cv2.goodFeaturesToTrack()
        p0 = cv2.goodFeaturesToTrack(Apad, mask = None, **feature_params)
        print len(p0)


        #for each harris corner point we compuete flow around 3*3 matrix

        for pt in range(0,len(p0)) :
        #for i in range(halfwindow,Apad.shape[0]-halfwindow):
            #for j  in range(halfwindow,Bpad.shape[1]-halfwindow):
                i = p0[pt][0][1]
                j = p0[pt][0][0]
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

                #pine

                us[i][j] = C[0]
                vs[i][j] = C[1]

        #stor the flow after iteration to u and v
        u[:] = u+us[halfwindow:Apad.shape[0]-halfwindow,halfwindow:Apad.shape[1]-halfwindow]
        v[:] = v+vs[halfwindow:Apad.shape[0]-halfwindow,halfwindow:Apad.shape[1]-halfwindow]



    uv = []
    # count=0
    # count2 = 0
    # for i in range(0,u.shape[0]):
    #     for j in range(0,u.shape[1]):
    #         if u[i][j] !=0:
    #             count+=1
    #         if v[i][j] !=0:
    #             count2+=1
    # print count
    # print count2
    uv.append(u)
    uv.append(v)
    return uv


imList1 = ['basketball1.png','teddy1.png','grove1.png']
imList2 = ['basketball2.png','teddy2.png','grove2.png']
# image1 =  'basketball1.png'
# image2 = 'basketball2.png'
windowsize = 5
halfwindow  = int(windowsize/2)



#lucas kanade step will be peerformed for each of image pairs

for im_iter in range(0,3):

    I1 = Image.open(imList1[im_iter]).convert('L')
    I2 = Image.open(imList2[im_iter]).convert('L')



    A = np.array(I1)
    B = np.array(I2)

    #compute pyramid for image A and image B



    res = calcLK(I2,A,B,halfwindow,0.001)

    u = np.zeros((A.shape[0],A.shape[1]))
    v = np.zeros((A.shape[0],A.shape[1]))
    u[:] = res[0]
    v[:] = res[1]

    spacing  = 10
    print "a shape is",A.shape
    print "u spahe is",u.shape
    print "v spahe is",v.shape

    Y, X = np.mgrid[0:A.shape[0], 0:A.shape[1]]

    # u1 = u[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    # v1 = v[0:v.shape[0]:spacing,0:v.shape[1]:spacing]
    #
    # x1 = X[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    # y1 = Y[0:u.shape[0]:spacing,0:u.shape[1]:spacing]
    plt.figure(im_iter*2+0)
    plt.imshow(I1)
    plt.gray()
    #
    # speed = np.sqrt(u1*u1 + v1*v1)
    #
    #d
    # lw = 10*speed/speed.max()
    #
    # print "x1 shape is",x1.shape
    # print "u1 shape is",u1.shape

    #plt.streamplot(X, Y, u1, v1, color=u1, linewidth=lw, cmap=plt.cm.autumn)
    #plt.streamplot(X, Y, u, v, density=[0.5,1])

    #plotting optical flow in image 1
    for i in range(0,u.shape[0]):
        for j in range(0,u.shape[1]):
            if(u[i][j]!=0.0 or v[i][j]!=0.0):
                plt.arrow(j,i,u[i][j],v[i][j],color = 'red',head_width = 6,head_length = 7)

    #plotting  optical flow on image2
    # plt.figure(im_iter*2+1)
    # plt.imshow(I2)
    # plt.gray()
    # for i in range(0,u.shape[0]):
    #     for j in range(0,u.shape[1]):
    #         if(u[i][j]!=0.0 or v[i][j]!=0.0):
    #             plt.arrow(j,i,u[i][j],v[i][j],color = 'yellow',head_width = 6,head_length = 7)



#plt.colorbar()

plt.show()

