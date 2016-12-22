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

#########
    #guass mean shift cluster is performed as follows steps:
    #R G B channels of Image is divided into 3 arrays
    #weight martix table is compueted before to look of guassian weights of that particualr intensity ievels
    #image id padded in order to process border pixels correctly
    #for each time
            # step 1 for each pixel of the image an attractio basin of hs * hs is defined and the each pixels value is multiplied by its guassian weight values
            #and the cumulative weights is added
            #and probability of each pixels belonging to each pixels with in that attraction  basin is caluated
            #each pixel(product * intensity value) is divided by cumulative weight to get the normalized value
            #this value is subtracted fromoriginal  intensity value to get the sift vector
            #again from step 1 the entore prcess is continued untill the sfit is less than threshold

#Programming consideration
    #as the algothim is very resource intensive and serialed version suffers for computation time ,hence from step1 onward each calualtion is done for the entire array at one which helps to speed up the process

#output
    #program will run through 67 intreation where  shift becomes  will is less than or equal to threshold and final image is display.

#########
#about threshold value
#threshold values hs and hr 20 and 32 will take around 67 iteration but converge or spread toward s more pixels  to converge while hs  = 8 and hs 7 tries to converge in few iterations

threshold = 0.01
hs = 20
hr = 32
# hs= 8
# hr = 7


#load the input color truth image
image_name2 = 'input3.jpg'
i1 = cv2.imread(image_name2)

I1 = np.array(image_name2)
# plt.figure(0)
# plt.imshow(i1)

B,G,R = cv2.split(i1)


#pad the image for processing the border pixels properly
#image is padded with size of image width on left and right side and with image height on top and bottom symmetrically.
R_padded= np.pad(R,((R.shape[0],R.shape[0]),(R.shape[1],R.shape[1])),'symmetric')
G_padded= np.pad(G,((G.shape[0],G.shape[0]),(G.shape[1],G.shape[1])),'symmetric')
B_padded= np.pad(B,((B.shape[0],B.shape[0]),(B.shape[1],B.shape[1])),'symmetric')



#frame the look up table to see correspoding guassian weight value for each (x-xn)*(x-xn) value
w_table = np.linspace(0, 256*256, num=256*256+1)


val1 = hr*hr
t1 = np.divide(w_table,val1)

w_table = np.exp(-t1)
print ("for hs = 20 and hr = 32 itis around 67 iterations to converge to the threshold")
iter = 0
while True:

    iter+=1
    #print(iter)
    total_denominator = np.zeros(R.shape,dtype=int);
    total_numerator_R = np.zeros(R.shape,dtype=int);
    total_numerator_G = np.zeros(R.shape,dtype=int);
    total_numerator_B = np.zeros(R.shape,dtype=int);

    difference_square_R = np.zeros(R.shape,dtype=int)
    difference_square_G = np.zeros(R.shape,dtype=int)
    difference_square_B = np.zeros(R.shape,dtype=int)

    for i in range(-hs,hs):
        for j in range(-hs,hs):
            #i = -hs to hs and j = -hs to hs is the attraction basin of each pixel value of R G and B array
            #get the array for this spatial range of i and j
            #i.e., each pixel in the array will be looked around its window of width hs and height hs
            R_current_region = R_padded[R.shape[0]+i-1:(2*R.shape[0])+i-1,R.shape[1]+j-1:(2*R.shape[1])+j-1]
            G_current_region = G_padded[G.shape[0]+i-1:(2*G.shape[0])+i-1,G.shape[1]+j-1:(2*G.shape[1])+j-1]
            B_current_region = B_padded[B.shape[0]+i-1:(2*B.shape[0])+i-1,B.shape[1]+j-1:(2*B.shape[1])+j-1]

            #compute the distance square array of this to the orginal R array



            difference_square = np.multiply((R-R_current_region),(R-R_current_region))
            difference_square_R = np.multiply((R-R_current_region).astype(int),(R-R_current_region).astype(int))
            difference_square_G = np.multiply((G-G_current_region).astype(int),(G-G_current_region).astype(int))
            difference_square_B = np.multiply((B-B_current_region).astype(int),(B-B_current_region).astype(int))

            #to avoid minimum value of zero for each R,G,B array
            difference_square_R = difference_square_R+1
            difference_square_G = difference_square_G+1
            difference_square_B = difference_square_B+1

            #get the kernel weight from w_table
            weight_map_R = w_table[difference_square_R]
            weight_map_G = w_table[difference_square_G]
            weight_map_B = w_table[difference_square_B]

            #compute the product of 3 weight matrices since they are  e (x-xn)*(x-xn) +(y-yn)*(y-yn)+(z-zn)*(z-zn) =e (x-xn)*(x-xn) * e(y-yn)*(y-yn)* e(z-zn)*(z-zn)

            temp = np.multiply(weight_map_R,weight_map_G)
            final_weight = np.multiply(temp,weight_map_B)


            total_denominator = total_denominator+final_weight

            total_numerator_R = total_numerator_R+(R_current_region*final_weight)
            total_numerator_G = total_numerator_G+(G_current_region*final_weight)
            total_numerator_B = total_numerator_B+(B_current_region*final_weight)

    final_value_R  = np.divide(total_numerator_R,total_denominator)
    final_value_G  = np.divide(total_numerator_G,total_denominator)
    final_value_B  = np.divide(total_numerator_B,total_denominator)

    shifted_vector_R = np.round(final_value_R)-np.round(R)
    shifted_vector_G = np.round(final_value_G)-np.round(G)
    shifted_vector_B = np.round(final_value_B)-np.round(B)

    R = np.round(final_value_R)
    G = np.round(final_value_G)
    B = np.round(final_value_B)

    #if the total mean shift of the final R,G,B array values is less than threshold then break the while loop

    mean_value = np.mean(shifted_vector_R+shifted_vector_G+shifted_vector_B)
    if abs(mean_value)<threshold:
        break;
    # output_image = np.zeros([R.shape[0],R.shape[1],3])
    # output_image[:,:,0] = B
    # output_image[:,:,1] = G
    # output_image[:,:,2] = R
    # plt.figure(1)
    plt.title('iteration'+str(iter))
    # plt.imshow(output_image)
    # plt.show()
    # cv2.imshow('imagewindow',output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print('iteration no is ',iter)
    print('shifted mean value in this iteraton',abs(mean_value))



#frame the  colour image array from individual R G B array
output_image = np.zeros([R.shape[0],R.shape[1],3])
output_image[:,:,0] = B
output_image[:,:,1] = G
output_image[:,:,2] = R

#misc.pilutil.imshow(output_image)
plt.imshow(output_image)
plt.show()



