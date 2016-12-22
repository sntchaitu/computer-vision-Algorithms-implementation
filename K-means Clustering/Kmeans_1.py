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
# from scipy import interpolate
#import cv2
# import scipy as sp
from matplotlib.pyplot import hist
# from sklearn.cluster import  KMeans
"""
Below program implements k-means clustering algorithms as below steps

1) initially select K centroids data points(x,y) which are farthest from each other


2) assign each pixel  of the image except the centroid points to one of those centroid bins based on the difference in the euclidean distance between that centroid and data point

3) Now compuete the actualcentroid point of the cluster using the below formula

Xc = 1/M(sigma(i = 1 to m) (xi)*(mi))
where xi is the x co-ordinate of the pixel, mi is the intensity of the pixel value
and n is the no of pixels of that cluster
M is the sum of the intensity values of the cluster

4)Now if the centroid found in the step 3 is different from step2 perform again step 2 and step3 iteratively.Else stop

5)Now resepresent each region with seperate colour
"""


image_name1 = 'input1.jpg'
i1 = Image.open(image_name1).convert('L')
#i3 = cv2.imread(image_name1)
I1 = np.array(i1,dtype = np.float32)

#load output image
image_name2 = 'out1.jpg'
i2 = Image.open(image_name2).convert('L')
#i3 = cv2.imread(image_name1)
I2 = np.array(i2,dtype = np.float32)

no_of_clsuters = 2

##########################################################################k means start###############################################################

#get the minimum intensity value of the array I1
minimum = np.min(I1)
#convert a 2d array to a id array
I1_vector = np.reshape(I1,I1.shape[0]*I1.shape[1])
#to eliminate zero intensity values
I1_vector = I1_vector-minimum+1
#create multilple copy of  the 1-d array for K times
I1_vector = np.tile(I1_vector,[no_of_clsuters,1])
#compute the initial center of mean for K  clusters
#[m1,m2,m3.............mk]
# each mean value = cluster number*maximumintensityvalue(I1_vector)/(K+1)

#v =  [value for index,value in enumerate(np.transpose(non_zero_indices))  if (I1[value[0]][value[1]]<mean_intensity_value+threshold) and (I1[value[0]][value[1]]>mean_intensity_value-threshold)]
#vec_mean=(1:K).*max((vector))/(no_of_cluster+1);
maximum = np.max(I1_vector)

mean_value = np.zeros((1,no_of_clsuters))

mean_value[:] = [value*np.max(I1_vector)/(no_of_clsuters+1)   for index,value in  enumerate(range(1,no_of_clsuters+1))]

vector_length = len(I1_vector[0])


#no of iterations
iter  = 0
old_mean_value  = np.zeros(mean_value.shape)
minimum_indices = np.zeros(I1_vector.shape)
while True:
    iter = iter+1
    #duplicate the mean_value 1d array until  the length of I1_vector

    #store mean value to old mean value
    old_mean_value[:] = mean_value

    #duplicate the mean_value until the length of I1_vector thi is used to compute minimum of euclidean distance
    mean_value = np.tile(mean_value,[vector_length,1]);
    mean_value = mean_value.transpose()

    #compute square of euclidean distance between that centroid and data point
    t1 = I1_vector-mean_value
    dis =  np.multiply((I1_vector-mean_value),(I1_vector-mean_value))

    #get the index of minimum values across the array
    minimum_indices = np.argmin(dis,axis=0)

    #resize the mean_value to single instance

    mean_value = mean_value[0:no_of_clsuters,0:1]

    #loop through the I1_vector for no_of cluster times to compute the actual mean of each cluster
    for i in range (0,no_of_clsuters):
        #index = np.where(minimum_indices is i)
        m1=minimum_indices==i
        no_of_values =  np.count_nonzero(m1)
        m1 = ~m1
        I3 = np.ma.masked_array(I1_vector[0],mask =m1)
        mean_value[i,:] = np.sum(I3)/no_of_values
    mean_value = mean_value.transpose()
    if(np.array_equal(mean_value,old_mean_value) or iter>25):
        break;
label_im = np.zeros(I1.shape)
label_im[:]= np.reshape(minimum_indices,I1.shape)

#display the image
plt.figure(2)
plt.imshow(label_im)
plt.show()

#print "here"


############################################################################k-means ends#####################################################################
########################calculation of TP rate FP rate and F score##############################################################

#conventions for this clustering image

#label_im arrays is the finalimage after clustering
#it contains values 0.0  which means background not tiger
#it contians values 1.0  foreground which means tiger

#conventions for this out2.jpg ground truth   image

#it contains values 0.0  which means foregorunf   tiger
#it contians values 255.0  means background  which means not tiger



# TP = count of no of values which are true in both GT array and in seed array label_im (GT intersection Actual)
tp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 0.0 and label_im[i][j] == 1.0:
            tp_count+=1


#print tp_count
# FP = count of no of values which are true in seed_Array(label_im) and false in GT(I2)(not GT intersection Actual)
fp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 255.0 and label_im[i][j] == 1.0:
            fp_count+=1


# FN = count of no of values which are false in label_im and true in GT  I2(GT intersection not Actual)
fn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 0.0 and label_im[i][j] == 1.0:
            fn_count+=1


# TN = Count of no of values which are false in label_im and false in GT I2(not GT intersection not Actual)
tn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 255.0 and label_im[i][j] == 0.0:
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


print"Although k means clustering gives bearable result but current image has  intenisty levels across the image that are not homogeneous." \
     "current image image has lot of intensity variation with in small spatial range hence this method also suffers to some extent "