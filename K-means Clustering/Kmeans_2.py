from __future__ import division
import Queue
from imaplib import _Authenticator
from sys import  argv
from PIL import Image
from PIL import ImageChops as ic
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
# from  scipy.ndimage import filters
# from scipy import misc
# from scipy import  signal
import math
from numpy.linalg import inv
import copy as cp
from numpy.linalg import inv
from scipy import interpolate
import cv2
import scipy as sp
from matplotlib.pyplot import hist
from sklearn.cluster import  KMeans
from numpy import NaN, Inf, arange, isscalar, asarray, array
"""
Below program implements k-means clustering algorithms as below steps for color image

6 cluster  initial cluster centers are choosen randomly from the values of histogram

1)seperate out the R G B intensity values of the image into 3 seperate vectors and the distance fucntion = sqrt(green*red+blue*blue+gree*green)

1) initially select K centroids data points(x,y) based on the histogram of the image R G B value vectors


2) assign each pixel  of the image except the centroid points to one of those centroid bins based on the difference in the euclidean distance between that centroid and data point

3) Now compuete the actualcentroid point of the cluster using the below formula

Xc = 1/M(sigma(i = 1 to m) (xi)*(mi))
where xi is the x co-ordinate of the pixel, mi is the intensity of the pixel value
and n is the no of pixels of that cluster
M is the sum of the intensity values of the cluster

4)Now if the centroid found in the step 3 is different from step2 perform again step 2 and step3 iteratively.Else stop

5)Now resepresent each region with seperate colour
"""






image_name1 = 'input2.jpg'
i1 = Image.open(image_name1)

I1 = cv2.imread(image_name1)


#load output image
image_name2 = 'out2.jpg'
i2 = Image.open(image_name2).convert('L')
#i3 = cv2.imread(image_name1)
I2 = np.array(i2,dtype = np.float32)

no_of_clsuters = 6

##########################################################kmeans++#########################################################

def kmeansplus(I1):

    #IN k means++ algorithm  cluster centers are initially chosen at random from the set of input observation vectors,
    # where the probability of choosing  a intensity level is high if x is not near any previously chosen intensity level centers.
    #x will be the next centre only if  prob( ||c1-x||^2/ sum x = 1 to N ||c1-x||^2) is maximum



    #compute histogram of the image and choose 2 points from fathest intesity level
    hist, bins = np.histogram(I1.ravel(),256,[0,256])

    #sort the hist based on intensity levels
    h1 = np.zeros(hist.shape,dtype=int)
    h1[:] = np.sort(hist)


    #get the unique intensiyt levels of the histogram
    l1 = []
    intensity1 = []
    #print h1
    for i in range(0,256):
        if hist[i] != 0  and hist[i] not in l1:
            l1.append(hist[i])
            intensity1.append(int(bins[i]))

    #for debuggig purppose
    #print intensity1




    #choose first data point as first centre
    C = [intensity1[0]]

    #loop untill required centres are obtained
    for k in range(0,no_of_clsuters-1):
        #each time compuete the minimum ditance between each probable centre point and already obtained centre points
        D2 = np.array([min([np.inner(c-x,c-x) for c in C]) for x in intensity1])

        probs = D2/D2.sum()
        #cumulative sum of probabilites of all centre points
        cumprobs = probs.cumsum()

        #generate a random number between 0 and 1
        r = np.random.rand()
        #if the cumulative probability for that pth point is less than r then select it as new random centre
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(i)
        #C.append(intensity1[probs.argsort().max()])
    return C




##############################################################kmeans++ ends##################################################



R,G,B = cv2.split(I1)
#miniumum value of R channel
#R_vector = np.reshape(R,R.)
r_vector = np.reshape(R,R.shape[0]*R.shape[1])
g_vector = np.reshape(G,G.shape[0]*G.shape[1])
b_vector = np.reshape(B,B.shape[0]*B.shape[1])

#miniumum value of R,G,B  channel

minR = np.min(r_vector)
minG = np.min(g_vector)
minB = np.min(b_vector)
#subtract min value from r_vector
r_vector = r_vector-minR+1
g_vector = g_vector-minG+1
b_vector = b_vector-minB+1


#create multilple copy of  the 1-d array for K times
r_vector = np.tile(r_vector,[no_of_clsuters,1])
g_vector = np.tile(g_vector,[no_of_clsuters,1])
b_vector = np.tile(b_vector,[no_of_clsuters,1])



mean_red = np.zeros((1,no_of_clsuters))
mean_green = np.zeros((1,no_of_clsuters))
mean_blue = np.zeros((1,no_of_clsuters))

# mean_red[:] = kmeansplus(r_vector)
# mean_green[:] = kmeansplus(g_vector)
# mean_blue[:] = kmeansplus(r_vector)

mean_red[:] = [value*np.max(r_vector)/(no_of_clsuters+1)   for index,value in  enumerate(range(1,no_of_clsuters+1))]
mean_green[:] = [value*np.max(g_vector)/(no_of_clsuters+1)   for index,value in  enumerate(range(1,no_of_clsuters+1))]
mean_blue[:] = [value*np.max(b_vector)/(no_of_clsuters+1)   for index,value in  enumerate(range(1,no_of_clsuters+1))]


vector_length = len(r_vector[0])


#no of iterations
iter  = 0
old_mean_red  = np.zeros(mean_red.shape)
minimum_indices = np.zeros(r_vector.shape)

old_mean_green  = np.zeros(mean_green.shape)
minimum_indices_g = np.zeros(g_vector.shape)

old_mean_blue  = np.zeros(mean_blue.shape)
minimum_indices_b = np.zeros(b_vector.shape)



while True:
    iter = iter+1
    #duplicate the mean_value 1d array until  the length of r_vector

    #store mean value to old mean value
    old_mean_red[:] = mean_red


    #duplicate the mean_value until the length of I1_vector thi is used to compute minimum of euclidean distance
    mean_red1 = np.tile(mean_red,[vector_length,1]);
    #mean_red = mean_red.transpose()


    #duplicate the mean_value until the length of I1_vector thi is used to compute minimum of euclidean distance
    mean_green1 = np.tile(mean_green,[vector_length,1]);
    #mean_green = mean_green.transpose()


    #duplicate the mean_value until the length of I1_vector thi is used to compute minimum of euclidean distance
    mean_blue1 = np.tile(mean_blue,[vector_length,1]);
    #mean_blue = mean_blue.transpose()


    mean_red1= mean_red1.transpose()
    mean_green1= mean_green1.transpose()
    mean_blue1= mean_blue1.transpose()

    #compute square of euclidean distance between that centroid and data point
    disr =  np.multiply((r_vector-mean_red1),(r_vector-mean_red1))
     #compute square of euclidean distance between that centroid and data point
    disg =  np.multiply((g_vector-mean_green1),(g_vector-mean_green1))
    #compute square of euclidean distance between that centroid and data point
    disb =  np.multiply((b_vector-mean_blue1),(b_vector-mean_blue1))

    distance = np.sqrt(disr+disg+disb)



    #get the index of minimum values across the array
    minimum_indices = np.argmin(distance,axis=0)


    for i in range(0,no_of_clsuters):
        index=(minimum_indices==i)
        index = ~index

        r_true = np.ma.masked_array(r_vector[0],mask =index)

        g_true = np.ma.masked_array(g_vector[0],mask =index)

        b_true = np.ma.masked_array(b_vector[0],mask =index)

        mean_red[:,i]     = np.ceil(np.mean(r_true))
        mean_green[:,i]   = np.ceil(np.mean(g_true))
        mean_blue[:,i]    = np.ceil(np.mean(b_true))


    # mean_red = mean_red.transpose()
    # mean_green = mean_green.transpose()
    # mean_blue= mean_blue.transpose()

    if ((np.array_equal(mean_red,old_mean_red) and np.array_equal(mean_red,old_mean_red) and np.array_equal(mean_red,old_mean_red)) or iter>25):
        break;


label_im = np.zeros(R.shape)
# label_im[:]= np.reshape(minimum_indices,I1.shape)

label_im = np.reshape(minimum_indices,R.shape)
#vec_mean = [meanred1;meangreen1;meanblue1];

print np.unique(label_im)

#display the image
plt.figure(2)
plt.imshow(label_im)
plt.figure(3)
plt.imshow(I2)


# print I2[174][408]
# print I2[266][320]
# print I2[301][325]
# print I2[38][362]
# print I2[175][175]

#print np.unique(I2)

I3 = np.zeros(I2.shape)
I3[:] = I2[:]
############################################################################k-means ends#####################################################################

# #maunally quantize the values of output image to only 6 values
for i in range(0, I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if(I3[i][j]>=0 and I3[i][j]<=25 ):
            I3[i][j]= 2
        if(I3[i][j]>=26 and I3[i][j]<=75 ):
            I3[i][j]= 0
        if(I3[i][j]>=75 and I3[i][j]<=125 ):
            I3[i][j]= 1
        if(I3[i][j]>=126 and I3[i][j]<=175 ):
            I3[i][j]= 3
        if(I3[i][j]>=176 and I3[i][j]<=225 ):
            I3[i][j]= 4
        if(I3[i][j]>=226 and I3[i][j]<=255 ):
            I3[i][j]= 5

# I3 [I2==102.0] = 2
# I3 [I2==153.0] = 0
# I3 [I2==204.0] = 1
# I3 [I2==255.0] = 3
# I3 [I2==127.0] = 4
# I3 [I2==0.0]   = 5
#
#
plt.figure(4)
plt.imshow(I3)
plt.show()
# #conventions for the clustered image
# #we assume pixels belonging to owl and branch as foreground and rest are background
#
# # TP = count of no of values which are true in both GT array and in seed array label_im (GT intersection Actual)
tp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 0 and label_im[i][j] == 0 and I2[i][j]== 1 and label_im[i][j] == 1 and I2[i][j]== 3 and label_im[i][j] == 3 and I2[i][j]== 4 and label_im[i][j] == 4:
            tp_count+=1


#print tp_count
# FP = count of no of values which are true in seed_Array(label_im) and false in GT(I2)(not GT intersection Actual)
fp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if (label_im[i][j]== 0 or label_im[i][j] == 1 or label_im[i][j] == 3 or label_im[i][j] == 4) and (I2[i][j]== 2 or I2[i][j] == 5):
            fp_count+=1


# FN = count of no of values which are false in label_im and true in GT  I2(GT intersection not Actual)
fn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if (I2[i][j]== 0 or I2[i][j] == 1 or I2[i][j] == 3 or I2[i][j] == 4) and (label_im[i][j]== 2 or label_im[i][j] == 5):
            fn_count+=1


# TN = Count of no of values which are false in label_im and false in GT I2(not GT intersection not Actual)
tn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 2 and label_im[i][j] == 2 and I2[i][j]== 5 and label_im[i][j] == 5  :
            tn_count+=1


#TP rate = TP/TP+FN
tpr= tp_count/(tp_count+fn_count)
print "TPR is",tpr

#fp rate is
fpr= fp_count/(fp_count+tn_count)
print "fpr is",fpr

#print fpr
#F-score as 2TP/(2TP + FP + FN)
fscore = (2*tp_count)/((2*tp_count)+fp_count+fn_count)
print "Fscore",fscore

print"comments about the output: As the image has various levels of intensity with in small range K means clustering done based on intensity levels along will not give a proper segmented image" \
     "and final clustered image suffers against ground truth image"
