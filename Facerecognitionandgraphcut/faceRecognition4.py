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
import scipy as sp
from matplotlib.pyplot import hist
from sklearn.cluster import  KMeans
import  scipy.ndimage as ndi
from skimage.feature import local_binary_pattern
import  sys as s1
import os
import re

#program instructions
#Extranct the trng folder and test folder to the same directory of programfile

#Below program will run the face recognition module
#each time using4 different feature vectors
#intesity level
#integral image
#LBP
#feature vecotr combined all features

#below program will run the face recognition code as below step:
#computes LBP feature vector,Integral Image feature vector,Intensity level and cobined feature vector
#removes mean from the feature vector
#computes the eigen vectors
#computes projected training matrix
#takes all test images and compuets the feature vector and computes PCA and find the euclidiean distance betwen the two
#which evere training imahe it matces it will be recognized as that image



#computes the LBP of the image and frames the histogram of 8 X 8 blocks and concatenates into a single feature vector
def fastLBP(I2,radius,nPoints,noofregions):
    #compuete neighbouring points
    spoints = np.zeros((nPoints,2))
    for n in range(0,nPoints):

        spoints[n][0] = (-radius)*math.sin((2*np.pi*n)/nPoints)
        spoints[n][1] = (radius)*math.cos((2*np.pi*n)/nPoints)


    ysize,xsize = I2.shape

    miny=min(spoints[:,0])
    maxy=max(spoints[:,0])
    minx=min(spoints[:,1])
    maxx=max(spoints[:,1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey=math.ceil(max(maxy,0))-math.floor(min(miny,0))+1;
    bsizex=math.ceil(max(maxx,0))-math.floor(min(minx,0))+1;

    # Coordinates of origin (0,0) in the block
    origy=1-math.floor(min(miny,0))-1
    origx=1-math.floor(min(minx,0))-1



    # Calculate dx and dy;
    dx = xsize - bsizex;
    dy = ysize - bsizey;

    # Fill the center pixel matrix C.
    C = I2[origy:origy+dy+1,origx:origx+dx+1]
    #d_C = double(C);

    bins = math.pow(2,nPoints)

    # Initialize the result matrix with zeros.
    result=np.zeros((dy+1,dx+1))

    #Compute the LBP code image
    for i  in range(0,nPoints):
        y = spoints[i][0]+origy
        x = spoints[i][1]+origx
        # Calculate floors, ceils and rounds for the x and y.
        fy = math.floor(y)
        cy = math.ceil(y)
        ry = round(y)
        fx = math.floor(x)
        cx = math.ceil(x)
        rx = round(x)
        # Check if interpolation is needed.
        if (abs(x - rx) < 0.000001) and (abs(y - ry) < 0.000001 ):
        # Interpolation is not needed, use original x and y co-ordinates
            N = I2[ry:ry+dy+1,rx:rx+dx+1]
            D = (N >= C)*1
        else:
        # Interpolation needed
            ty = y - fy;
            tx = x - fx;

            # Calculate the interpolation weights.
            w1 = (1 - tx) * (1 - ty)
            w2 = (tx) * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty


            # Compute interpolated pixel values
            a1 = w1*I2[fy:fy+dy+1,fx:fx+dx+1]
            a2 = w2*I2[fy:fy+dy+1,cx:cx+dx+1]
            a3 = w3*I2[cy:cy+dy+1,fx:fx+dx+1]
            a4 = w4*I2[cy:cy+dy+1,cx:cx+dx+1]

            N = a1+a2+a3+a4

            #print C.shape

            D = (N >= C) * 1

        # Update the result matrix.
        v = math.pow(2,i)
        result = result+v*D
    # plt.gray()
    # plt.figure(1)
    # plt.imshow(result)
 #   plt.show()


    # #compute the histogram of the LBP image across 8*8 block and concatenate all the histograms
    windowsize_r = noofregions
    windowsize_c = noofregions
    val = []
    for r in range(0,result.shape[0]-windowsize_r, windowsize_r):
        for c in range(0,result.shape[0]-windowsize_c , windowsize_c):
            window = result[r:r+windowsize_r,c:c+windowsize_c]
            hist= np.histogram(window,2**nPoints,range =(0,2**nPoints),normed=True)[0]
            #print np.count_nonzero(hist)
            #val +=hist
            val = val+ np.ndarray.tolist(hist)
    lbp_vector = np.asarray(val)
    return lbp_vector
    # plt.gray()
    # plt.figure(4)
    # plt.plot(val,color = 'b')
    #plt.show()






#compuete pLBP pattern for each of image block
def computeLBP(I2,radius,nPoints,noofregions):


    dst_img = np.zeros((I2.shape[0]-2*radius,I2.shape[1]-2*radius))
    for n in range(0,nPoints):

        x = (-radius)*math.sin((2*np.pi*n)/nPoints)
        y = (radius)*math.cos((2*np.pi*n)/nPoints)

        #relative indices
        bx = math.floor(x)
        by = math.floor(y)
        tx = math.ceil(x)
        ty = math.ceil(y)

        #fp
        ty = y - by
        tx = x - bx

        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty

        #iterate through the image
        for i in range(radius,I2.shape[0]-radius):
            for j in range(radius,I2.shape[1]-radius):
                #calc interpolated value
                t = w1*I2[i+by][j+bx] + w2*I2[i+by][j+tx] + w3*I2[i+ty][j+bx] + w4*I2[i+ty][j+bx]
                dst_img[i-radius][j-radius] += ((t > I2[i][j]) or (I2[i][j] < s1.float_info.epsilon)) << n


    #compute the histogram for the LBP array


    # plt.gray()
    # plt.figure(1)
    # plt.imshow(dst_img)
    #plt.show()

#compuete gray levels of integral image
def compute_gray_int_levels(I2):
    i5 = np.zeros(I2.shape)
    intim = cv2.integral(I2)
    i5[:]  = intim[1:,1:]
    #intim = (255.0*intim)/np.max(intim)
    return i5.reshape(I2.shape[0]*I2.shape[1])
    # I_integral = np.zeros(I2.shape)
    # #I_integral[:] = I2[:]
    # for i in range(0,I2.shape[0]):
    #     for j in range(0,I2.shape[1]):
    #         if(i-1>-1 and i-1<I2.shape[0] and i>-1 and i<I2.shape[0]and j>-1 and j<I2.shape[1]):
    #             I_integral[i][j] += I_integral[i-1][j]
    #         if(i>-1 and i<I2.shape[0] and j-1>-1 and j-1<I2.shape[1]):
    #             I_integral[i][j] += I2[i][j-1]
    #         if(i-1>-1 and i-1<I2.shape[0] and j-1>-1 and j-1 <I2.shape[1]):
    #             I_integral[i][j] -= I_integral[i-1][j-1]
    #         if(i>-1 and i<I2.shape[0] and j>-1 and j<I2.shape[1]):
    #             I_integral[i][j] += I2[i][j]
    # print "hi"
    # plt.gray()
    # plt.imshow(Image.fromarray(I_integral))
    # plt.show
    #return I_integral
# compute gray levels of normal image
def compute_gray_levels(I2):
    return np.reshape(I2,I2.shape[0]*I2.shape[1])

def fr(a1,a2,a3,a4):
    #compute feature vectors of training images
    #load the training set of images
    #p1 = 'images/yalefaces/bushimages'
    # p1 = 'images/yalefaces/trng1'
    # p2 = 'images/yalefaces/test1'
    p1 = 'trng'
    p2 = 'test'
    rad = 1
    npts = 8*rad
    nregion = 16

    list1 = os.listdir(p1)
    counter = 0
    final_vector = []
    training_image_names = []
    training_images_number= []
    #load all the training images from the directory
    #store the labels of each image in seperate list

    regex = re.compile(r'\d+')
    print len(list1)
    for file in list1:
        #store the actual subject numbers in a seperate list
        training_images_number.append([int(x) for x in regex.findall(file)][0])
        training_image_names.append(file)
        image_name = p1+'/'+file
        i1 = Image.open(image_name).convert('L')
        I2 = np.array(i1)
        #computeLBP(I2,rad,npts,nregion)
        if a3==True:
            f1 = fastLBP(I2,rad,npts,nregion)
            f = f1
        elif a2==True:
            f2 = compute_gray_int_levels(I2)
            f = f2
        elif a1 == True:
            f3 = compute_gray_levels(I2)
            f = f3
        elif a4 == True:
            f1 = fastLBP(I2,rad,npts,nregion)
            f2 = compute_gray_int_levels(I2)
            f3 = compute_gray_levels(I2)
            f4 = np.zeros((1,f1.shape[0]+f2.shape[0]+f3.shape[0]))
            #concatenate 3 feature vectors into single feature vector for each image
            f4[:]=  np.concatenate((f1,f2,f3),axis=0)
            f = f4

        if counter == 0:
            counter+=1
            final_vector = np.zeros(f.shape)
        final_vector = np.vstack((final_vector,f))
    print "images size",len(training_images_number)

    #remove first row
    final_vector = np.delete(final_vector,0,0)

    mean1 = final_vector.mean(axis=0)
    final_vector = final_vector - mean1
    c = np.dot(final_vector,final_vector.transpose())
    [eigenvalues ,eigenvectors] = np.linalg.eigh(c)
    tmp = np.dot(final_vector.T,eigenvectors).T # this is the compact trick
    V = tmp[::-1] # reverse since last eigenvectors are the ones we want
    S = np.sqrt(eigenvalues)[::-1] # reverse since eigenvalues are in increasing order
    for i in range(V.shape[1]):
        V[:,i] /= S


    #take only 20 best eigen vectors
    v1 = V[0:20,:]

    #display forst 20 eigenfaces
    # plt.gray()
    # plt.figure(1)
    # plt.imshow(mean1.reshape(I2.shape))
    # plt.figure(2)
    # plt.imshow(V[0].reshape(I2.shape))
    # plt.show()

    #compute the dot product between the training image feature vector and V1
    #v1 is of dimensions 20 * 77760 and feature vector is of dimensions 117 * 77760

    training_project_martix = np.dot(v1,final_vector.T)

    #choose test images from the test folder
    list2 = os.listdir(p2)
    counter = 0
    final_vector1 = []
    testing_image_names = []
    testing_image_number = []

    for file in list2:
        testing_image_names.append(file)
        testing_image_number.append([int(x) for x in regex.findall(file)][0])
        image_name = p2+'/'+file
        i1 = Image.open(image_name).convert('L')
        I2 = np.array(i1)
        if a3==True:
            f1 = fastLBP(I2,rad,npts,nregion)
            f = f1
        elif a2==True:
            f2 = compute_gray_int_levels(I2)
            f = f2
        elif a1 == True:
            f3 = compute_gray_levels(I2)
            f = f3
        elif a4 == True:
            f1 = fastLBP(I2,rad,npts,nregion)
            f2 = compute_gray_int_levels(I2)
            f3 = compute_gray_levels(I2)
            f4 = np.zeros((1,f1.shape[0]+f2.shape[0]+f3.shape[0]))
            #concatenate 3 feature vectors into single feature vector for each image
            f4[:]=  np.concatenate((f1,f2,f3),axis=0)
            f = f4

        if counter == 0:
            counter+=1
            final_vector1 = np.zeros(f.shape)
        final_vector1 = np.vstack((final_vector1,f))
        #computeLBP(I2,rad,npts,nregion)
        # f1 = fastLBP(I2,rad,npts,nregion)
        # f2 = compute_gray_int_levels(I2)
        # f3 = compute_gray_levels(I2)
        # f[:]=  np.concatenate((f1,f2,f3),axis=0)
        # if counter == 0:
        #     counter+=1
        #     final_vector1 = np.zeros(f3.shape)
        # final_vector1 = np.vstack((final_vector1,f3))

    #remove first row
    final_vector1 = np.delete(final_vector1,0,0)
    #subtract the mean
    final_vector1 = final_vector1 - mean1

    testing_project_martix = np.dot(v1,final_vector1.T)
    #print "hi"
    # #compute the eculidian dsitance between each test matrix with training matrix
    distance_per_test_image = []
    index_list = []
    minValue = 1000000000
    for i in range(0,testing_project_martix.shape[1]):
        index_num = 0
        distance_per_test_image = []
        for j in range(0,training_project_martix.shape[1]):
            c1 = testing_project_martix[:,i]
            c2 = training_project_martix[:,j]
            distance_per_test_image.append(np.sum((testing_project_martix[:,i]-training_project_martix[:,j])**2))
        index_num = np.argmin(np.asarray(distance_per_test_image))
        c3 = training_images_number[index_num]
        index_list.append(c3)

    #print index_list
    count = 0
    #print testing_image_names
    #print testing_image_number
    for i in range(0,len(index_list)):
        if index_list[i]==testing_image_number[i]:
            count+=1
    if a1==True:
        print 'accuracy using Grey scale Intensity level as feature vector',(count/len(testing_image_number))*100

    if a2==True:
        print 'accuracy using Grey scale Integral Image  as feature vector',(count/len(testing_image_number))*100

    if a3==True:
        print 'accuracy using LBP as feature vector',(count/len(testing_image_number))*100

    if a4==True:
        print 'accuracy using all features combined as feature vector',(count/len(testing_image_number))*100



fr(True,False,False,False)

fr(False,True,False,False)

fr(False,False,True,False)

fr(False,False,False,True)




