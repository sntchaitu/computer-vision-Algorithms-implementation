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


#load the ground truth image
image_name2 = 'out1.jpg'
i1 = Image.open(image_name2).convert('L')
I2 = np.array(i1)



#load the input image
image_name1 = 'input1.jpg'
i1 = Image.open(image_name1).convert('L')
# #i3 = cv2.imread(image_name1)
I1 = np.array(i1,dtype = np.float32)
plt.imshow(i1)
plt.gray()
#display the image and ask the user to click for intial seed point
print 'Please click initial seed points'

seed_point = plt.ginput(1)
#print seed_point[0][0],seed_point[0][1]
#x = 230
#y = 130
initial_pixel = []
# initial_pixel.append(x)
# initial_pixel.append(y)
initial_pixel.append(int(seed_point[0][0]))
initial_pixel.append(int(seed_point[0][1]))
# initial_pixel.append(int(seed_point[0][0]))
# initial_pixel.append(int(seed_point[0][1]))
print 'you clicked:',initial_pixel

###################################region growing algorihm starts#########################################
#create seed array it is filled with 255 for those values that belong/similar to initial seed and rest are zero

seed_array = np.zeros(I1.shape)
seed_array[initial_pixel[0]][initial_pixel[1]] = 255.0
seed_copy = np.zeros(seed_array.shape)

seed_intensity = I1[initial_pixel[0]][initial_pixel[1]]
threshold = 25


struct2 = ndi.generate_binary_structure(2, 2)
dilation_res = np.zeros(I1.shape)

#until there is no change in seed array the loop continues that is there is no region to be added to the existing part of segmented image
while seed_array.sum()!=seed_copy.sum():

    seed_copy[:] = seed_array
    res =  (np.where(seed_array>0))
    mean_intensity_value = np.mean(I1[res[0],res[1]])
    dilation_res = ndi.binary_dilation(seed_array,struct2).astype(I1.dtype) -seed_array
    non_zero_indices = (np.where(dilation_res>0))
    #end = seed_array.shape[0]*seed_array.shape[1]+1
    #print (non_zero_indices)
    v =  [value for index,value in enumerate(np.transpose(non_zero_indices))  if (I1[value[0]][value[1]]<mean_intensity_value+threshold) and (I1[value[0]][value[1]]>mean_intensity_value-threshold)]
    if len(v)>0:
        v1= np.transpose(v).tolist()
        seed_array[v1[0],v1[1]] = 255.0
plt.imshow(Image.fromarray(seed_array))
plt.show()


########################calculation of TP rate FP rate and F score##############################################################



# TP = count of no of values which are true in both GT array and in seed array(GT intersection Actual)
tp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 255.0 and seed_array[i][j] == 255.0:
            tp_count+=1


print tp_count
# FP = count of no of values which are true in seed_Array and false in GT(not GT intersection Actual)
fp_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 0.0 and seed_array[i][j] == 255.0:
            fp_count+=1


# FN = count of no of values which are false in Actual and true in GT(GT intersection not Actual)
fn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 255.0 and seed_array[i][j] == 0.0:
            fn_count+=1


# TN = Count of no of values which are false in D and false in GT(not GT intersection not Actual)
tn_count = 0
for i in range(0,I2.shape[0]):
    for j in range(0,I2.shape[1]):
        if I2[i][j]== 0.0 and seed_array[i][j] == 0.0:
            tn_count+=1


#TP rate = TP/TP+FN
tpr= tp_count/(tp_count+fn_count)
print "TPR is",tpr

#fp rate is
fpr= fp_count/(fp_count+tn_count)
print "F score is",fpr

#print fpr
#F-score as 2TP/(2TP + FP + FN)



fscore = (2*tp_count)/((2*tp_count)+fp_count+fn_count)
print "Fscore",fscore



print"Comments about the result: Region growing works best only for those images where is has atmost 2 intenisty levels across the image or the image intensity levels are homogeneous." \
     "But the current image image has lot of intensity variation with in small spatial range this method suffers for this image"











#np.reshape(dilation_res,dilation_res.shape[0]*dilation_res.shape[1])
#print end
#seed_array = np.arange(1,end).reshape((seed_array.shape[0],seed_array.shape[1]))

#print non_zero_indices
# print non_zero_indices
# values = I1[non_zero_indices[0],non_zero_indices[1]]
# #print I1[non_zero_indices[0],non_zero_indices[1]]>(mean_intensity_value-threshold)
# #print I1[non_zero_indices[0],non_zero_indices[1]]<(mean_intensity_value+threshold)
# #print (values>mean_intensity_value-threshold)
# #non_zero_indices = np.transpose(non_zero_indices)
#
# #print seed_array[non_zero_indices[(values>mean_intensity_value-threshold)]]
# #print np.where(I1[non_zero_indices[0],non_zero_indices[1]]<(mean_intensity_value+threshold))
# #printI1


# for i in range (0,len(v)):
#     print v[i][0],v[i][1]
#     seed_array[v[i][0]][v[i][1]] = 255.0


#seed_array = np.arange(1,end).reshape((seed_array.shape[0],seed_array.shape[1]))
#seed_array[v1[0],v1[1]] = 1
#print v.tolist()
# print v1[1]
#print v1.tolist()[0],v1.tolist()[1]
# rows = v1.tolist()[0]
# cols = v1.tolist()[1]
# print rows,cols
# seed_array[v1.tolist()[0],v1.tolist()[1]] = 1

# a = np.array([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
# # a = np.arange(1,10).reshape((3,3))
# rows = [0,1]
# print rows
# cols= [2,2]
#
# a[rows,cols] = 0
# print a
#seed_array[v1[0],v1[1]] = 1.0
#v1 =  v1.transpose()
#print v1
#seed_array[v1] = 1.0
#print [index for index in enumerate(np.transpose(non_zero_indices)) if I1[index[0][0][0]][index[0][0][1]] < mean_intensity_value+threshold]
#print [non_zero_indices for index in enumerate(np.transpose(non_zero_indices)) if I1[index[0]][index[1]] < mean_intensity_value+threshold and I1[index[0]][index[1]] >mean_intensity_value-threshold]

#print indices
#seed_array[[non_zero_indices[0],non_zero_indices[1]][values>mean_intensity_value-threshold]] = 1.0
#print np.where(I1[non_zero_indices[0],non_zero_indices[1]]>(mean_intensity_value-threshold))
#np.mean(I1[res[0],res[1]])
#print res[0]
#print I1[res[0],res[1]]

#print struct2

#a1  =  np.zeros(I1.shape,dtype=bool)
# a2  =  np.zeros((3,3),dtype=bool)
# a3  =  np.zeros((3,3),dtype=float)
# a2[1][1] = True
#print a2



#a1[initial_pixel[0]][initial_pixel[1]] = True
#while a1.sum()!=a2.sum():


#print dilation_res
#print a3(dilation_res)
#print  a1(np.transpose(np.where(dilation_res>0)))


#x = np.arange(10,1,-1)
#print a3
#print a3[res]

#print a3[np.array([3, 3, 1, 8])]


#print a1
#display the output image
# plt.imshow(Image.fromarray(seed_array))
# plt.show()

