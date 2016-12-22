from __future__ import division

import Queue
#from scipy import misc
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

import math


"""
    Below program will all the required performance evaluation metrics for below
    Input Images used are:
    a)boat.jpg
    b)aeroplane.jpg
    c)pot.jpg


"""

from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math



"""

Below program will compute the below mentioned metrics on 3
edge images which are obtained after implementing canny edge detection
technique on them.

First we generate a boolean array for Ground truth image (GT) and for image obtained after applying canny edge detection
technique(Actual)
If the pixel value in  image is an edge i.e., if its pixel value is non zero then corresponding value in boolean array
is true else value is false




    TP = count of no of values which are true in both GT array and in Actual array(GT intersection Actual)
    FN = count of no of values which are false in Actual and true in GT(GT intersection not Actual)
    FP = count of no of values which are true in D and false in GT(not GT intersection Actual)
    TN = Count of no of values which are false in D and false in GT(not GT intersection not Actual)

Next
    we compuete below metrics

1)sensitivity  = TP/(TP+FN)
2)specificity = TN/(TN+FP)
3)Precision = TP/(TP+FP)
4)Negative Predictive value TN/(TN+FN)
5)Fall-out = FP/(FP+TN)
6)False Negative Rate  = FN/(FN+TP)
7)False Discovery Rate  = FP/FP+TP
8)Accuracy (TP+TN)/(TP+FN+TN+FP)
9)F-Score (2TP)/(2TP+FP+FN)
10)Matthews correlation (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))


for below images
    input_image.jpg
    input2_image.jpg
    output_image.jpg
    output_image2.jpg
"""

############################################################################################################################
# Below module is  for detecting canny edges a duplicate of Q1.py file
############################################################################################################################

def g_mask(sigma):
    """
    As The mask size should not be too large compared as it will result in unnecessary computational .
    At the same time, the mask size should not be so small to loose the characteristics  .
    Hence we keep see the Gaussian and applying a threshold T.T is a real-number between 0 and 1.
    First, size of half mask, radius is computed by finding the point on the curve where the Gaussian value drops below T
     T. By trail and error at T = 0.05 it gives good response for the input image
    , i.e. solving exp(-x^2/(2*sigma^2) = T. This gives radius = round(sqrt(-log(T) * 2 * sigma^2)).
     The mask size is then 2*radius + 1 to incorporate both positive and negative sides of the mask.
    :param sigma:
    :return:a list
            list[0] = guassian kernel mask
            list[1] = guassian derivative kernel
            list[2] = width of the kernel
    """

    width = 2*(int)(round(math.sqrt(-math.log(0.05)*2.0*(sigma*sigma))))+1
    #width = 6 * sigma +1
    #print width
    radius  = (int)(width/2)
    guassian_kernel = np.ones((1, width), np.float32)
    guassian_der = np.ones((1, width), np.float32)
    kernels = []
    total = 0
    # print width
    # In below loop we generate guassian mask G and guassian derivative mask for Gx x direction convolution.
    # for Gy = Transpose(Gx)
    # for G in y direction = Transpose(G)
    for i in range(-radius,radius+1):
        # at each interval i we compute the mean of i , i-0.5, i+0.5 and store it in guassian_kernel of i
        # compared to only value (i) this mean value will give good result after convolution
        g1 = gaussian(i,sigma)
        # if (g1 <= 0.005 and  i >= 2):
        #     break
        g2 = gaussian(i-0.5,sigma)
        g3 = gaussian(i+0.5,sigma)
        #guassian_der[0][i] = g3 - g2
        #guassian_kernel[0][i+radius] =  1/(math.sqrt(2*math.pi)*sigma) * gaussian(i, sigma)

        guassian_der[0][i+radius] =  (-i/(sigma*sigma)) * gaussian(i, sigma)
        #guassian_der[0][i+radius] =  (-i/(math.sqrt(2*math.pi)*sigma*sigma*sigma)) * gaussian(i, sigma)
        #guassian_der[0][i+radius] =  (-i/(sigma*sigma)) * gaussian(i, sigma)
        #guassian_kernel[0][i+radius] = (g1)*(1/(2.0 * math.pi * sigma * sigma))

        guassian_kernel[0][i+radius] = ((g1 + g2 + g3)/3.0)*(1/(math.sqrt(2.0 * math.pi * sigma * sigma)))
        #total += guassian_kernel[0][i+radius]
    # for i in range(-radius,radius+1):
    #     guassian_kernel[0][i+radius] = guassian_kernel[0][i+radius]/total

    #append the guassian, derivative fo guassian and guassian radius to the kernel
    kernels.append(guassian_kernel)
    kernels.append(guassian_der)
    kernels.append((int)(width/2))
    return kernels


def gaussian(x,sigma):
    """
    computes the guassian function value for x and sigma
    :param x:
    :param sigma:
    :return:value of guasian function for input x and sigma
    """

    if sigma == 0:
        return 0
    else:
        return math.exp(-(x * x) / (2.0 * sigma * sigma))



def flip(M,x):

    """
    This function checks  boundary values if value < 0 and value > M then returns the flipped position of that index value

    :param M:
    :param x:
    :return:
    """
    if(x<0):
        return -x-1
    elif x >= M:
        return 2*M-x-1
    else:
        return x


def canny(image_name,sigma,u_treshold,l_treshold):
    """
    computes the canny edges by ting sigma , u_treshold and l_treshold
    sigma value 1 the edge map is better than for other sigma values
    other better sigma values are 1.2 and 1.4
    :param sigma:
    :param u_treshold:
    :param l_treshold:
    :return:
    """
    image_list   = []
    image_description = []
    im = Image.open(image_name).convert('L')
    I0 = np.array(im)

    #I0 = np.ones((10,10))
    # stores the array after gaussian blurring in x direction
    Ix = np.zeros(I0.shape)

    # stores the array after gaussian blurring in y direction
    Iy = np.zeros(I0.shape)

    # stores the array after convolution with gaussian derivative in x direction
    Ix1 = np.zeros(I0.shape)


    # stores the array after convolution with gaussian derivative in x direction

    Iy1 = np.zeros(I0.shape)

    # stores the gradient of the image
    M = np.zeros(I0.shape)
    # guassian Kernel parameters
    #sigma = 1
    MAX_SIZE = 20

    # below calling function will return the guassian mask and derivative of gaussian mask and gaussian matrix radius
    kernels = g_mask(sigma)
    k1 = kernels[0]
    k2 = kernels[1]
    r1 = kernels[2]


# convolve the Image horizontally with kernel k1

    for y in range(0,I0.shape[0]):
            for  x in  range(0, I0.shape[1]):
                sum = 0.0;
                # for each pixel we convolve flipped gaussian kernel with the image window and stoes that value at x and y location
                for i in range(-r1 ,r1+1):
                    x1 = flip(I0.shape[1],x - i);
                    sum = sum + k1[0][i + r1]*I0[y][x1];
                Ix[y][x] = sum;

    # convolve the Image vertically with kernel Transpose(k1)
    for y in range(0,I0.shape[0]):
            for  x in range( 0, I0.shape[1]):
                sum = 0.0;
                for i in range(-r1 ,r1+1):
                    y1 = flip(I0.shape[0],y - i);
                    sum = sum + k1[0][i + r1]*I0[y1][x];
                Iy[y][x] = sum;

    # stores the image for display
    im01 = Image.fromarray(Ix)
    image_list.append(im01)
    image_description.append('Image after smoothing in x direction')
    # plt.figure(1)
    # plt.imshow(im01)


    im02 = Image.fromarray(Iy)
    image_list.append(im02)
    image_description.append('Image after smoothing in y direction')
    # plt.figure(2)
    # plt.imshow(im02)

    # convolve the Image horizontally with kernel k2
    for y in range(0,Ix.shape[0]):
            for  x in  range(0, Ix.shape[1]):
                sum = 0.0;
                for i in range(-r1 ,r1+1):
                    x1 = flip(Ix.shape[1],x - i);
                    sum = sum + k2[0][i + r1]*Ix[y][x1];
                Ix1[y][x] = sum;

    # convolve the Image vertically with kernel k2-Transpose
    for y in range(0,Iy.shape[0]):
            for  x in range( 0, Iy.shape[1]):
                sum = 0.0;
                for i in range(-r1 ,r1+1):
                    y1 = flip(Iy.shape[0],y - i);
                    sum = sum + k2[0][i + r1]*Iy[y1][x];
                Iy1[y][x] = sum;

    im03 = Image.fromarray(Ix1)
    image_list.append(im03)
    image_description.append('Image after applying guassian derivative in x direction')
    # plt.figure(3)
    # plt.imshow(im03)

    im04 = Image.fromarray(Iy1)
    image_list.append(im04)
    image_description.append('Image after applying guassian derivative in y direction')
    # plt.figure(4)
    # plt.imshow(im04)
    # # #plt.show()

    # below code block will compute magnitude and orientation value  of each pixel
    orientation = np.ones((Ix.shape[0], Ix.shape[1]), np.float32)
    M = np.ones((Ix.shape[0], Ix.shape[1]), np.float32)
    for i in range(0, Ix.shape[0]):
        for j in range(0, Iy.shape[1]):
            M[i][j] = math.sqrt(Ix1[i][j]*Ix1[i][j] + Iy1[i][j] * Iy1[i][j])
            #orientation[i][j] = math.degrees(math.atan2(Iy1[i][j], Ix1[i][j]))
            #orientation[i][j] = math.degrees(math.atan2(Iy1[i][j], Ix1[i][j]))

    im05 = Image.fromarray(M)
    image_list.append(im05)
    image_description.append('Gradient magnitude image')
    # plt.figure(5)
    # plt.imshow(im05)
    #


    # create a image of size M1 to store the non-maxima supressed values of entire image

    M1 = np.copy(M)

    # Compute the Ix/Ix
    # if Ix = 0 make tan value of that pixel to 5

    for i in range(1,M.shape[0]-1):
        for j in range(1,M.shape[1]-1):
            if (Ix1[i][j]==0):
                tan = 5
            else:
                tan = Iy1[i][j]/Ix1[i][j]
                # if the range is between -22.5 and 22.5 then those pixels falls with in zero degree bins
                # hence check of i,j+1 and i,j-1
                if (-0.4142<tan and tan<=0.4142):
                    if(M[i][j]<M[i][j+1]) or  M[i][j]<M[i][j-1]:
                        M1[i][j]=0;
                # if the range is between 22.5 and 67.5 and between 202.5 and 247.5 then those pixels falls with in 45 degree bins
                # hence check of i-1,j+1 and i+1,j-1
                if (0.4142<tan and tan<=2.4142):

                    if(M[i][j]<M[i-1,j+1] or M[i][j]<M[i+1,j-1]):

                     M1[i][j]=0

                # if the range is between 67.5 and 112.5 and between 180 + 67.5 and 292.5 then those pixels falls with in 90 degree bins
                # hence check of i-1,j and i+1,j
                if ( abs(tan) >2.4142):

                    if(M[i][j]<M[i-1,j] or M[i][j]<M[i+1,j]):

                        M1[i][j]=0;

                # if the range is between 67.5 and 112.5 and between 180 + 67.5 and 292.5 then those pixels falls with in 135 degree bins
                # hence check of i-1,j-1 and i+1,j+1
                if (-2.4142<tan and tan<= -0.4142):

                    if(M[i][j]<M[i-1][j-1] or M[i][j]<M[i+1][j+1]):

                        M1[i][j]=0;



    im07 = Image.fromarray(M1)
    image_list.append(im07)
    image_description.append('Image after non-max supression')
    # plt.figure(7)
    # plt.imshow(im07)
    # print M1
    # im7 = Image.fromarray(M)
    # plt.figure(7)
    # plt.imshow(im7)


    # below code block will perform hysterisis tresholding
    # if current pixel value in gradient array M is greater than upper treshold then we can consider that pixel value
    # as start of edge and go along the edge to see if the neighbouring pixel value is greater then lower treshold.
    # if it is greater than  lower threshold set the neighbouring pixel  value as 255 and continue to the next neighbour pixel along the edge
    # In the end supress all the pixels that are not 255 and grater than o then set those pixel values in M array 0
    q1 = Queue.Queue()
    # M1 = np.empty_like(M)
    # M1[:] = M

    for i in range(1,M1.shape[0]-1):
        for j in range(1,M1.shape[1]-1):
            if M1[i][j] > u_treshold:
                M1[i][j] = 255
                l = []
                l.append(i)
                l.append(j)
                q1.put(l)
                while not q1.empty():
                    x1, y1 = q1.get()
                    #add neighbours whose value is greater then lower threshold and less than upper threshold
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            if x1+m > -1 and y1+n > -1 and x1+m < M1.shape[0] and y1+n < M.shape[1] and M1[x1+m][y1+n] >= l_treshold and M1[x1+m][y1+n] < u_treshold :
                                M1[x1+m][y1+n] = 255
                                q1.put([x1+m,y1+n])
            else:
                M1[i][j] = 0

    #This block of code will make the image of only 2 levels all edge pixels are of intensity 255 and other pixels are of intensity 0
    for i in range(1,M1.shape[0]):
        for j in range(1,M1.shape[1]):
            if M1[i][j]>0:
                M1[i][j] = 255


    image_list.append(im07)
    image_description.append('Image after hysterisis')
    im8 = Image.fromarray(M1)
    # plt.figure(8)
    # plt.imshow(im8)
    # plt.show()

    # return final binaary image after canny edge detection
    return M1

############################################################################################################################
# canny edge detector module ends
############################################################################################################################





def compute_metrics(edge_map,computed_edge):
    """
    below function will compute all the necessary qualitative metrics for actual edge image and Ground Truth edge map
    for both input images
    First  we form GT array and actual array withonly True or false values.True means255 and false is 0.
    :param edge_map:
    :param computed_edge:
    :return:
    """
    GT = np.ones((edge_map.shape[0],edge_map.shape[1]),dtype=bool)
    actual = np.ones((edge_map.shape[0],edge_map.shape[1]),dtype=bool)
    count1 = 1;
    count2 = 1
    # below block will compute GT array
    for i in range (0,edge_map.shape[0]):
        for j in range(0,edge_map.shape[1]):
            if edge_map[i][j] ==0:
                GT[i][j] = False
            else:
                count1 += 1
    # below code block will compute actual array
    for i in range (0,edge_map.shape[0]):
        for j in range(0,edge_map.shape[1]):
            if computed_edge[i][j] ==0:
                actual[i][j] = False
            else:
                count2 += 1

    # below code will compute TP,FN,FP,TN values
    TP = 1
    TN = 1
    FN = 1
    FP = 1
    for i in range (0, edge_map.shape[0]):
        for j in range (0, edge_map.shape[1]):
            if GT[i][j]  and actual[i][j]:
                TP+=1
            elif  not GT[i][j]  and  not actual[i][j]:
                TN+=1
            elif  not GT[i][j]  and   actual[i][j]:
                FP+=1
            elif  GT[i][j] and  not actual[i][j]:
                FN+=1

    print "\nTP",TP
    print "\nTN",TN
    print "\nFP",FP
    print "\nFN",FN


    sensitivity  = TP/(TP+FN)
    specificity  = TN/(TN+FP)
    Precision    = TP/(TP+FP)
    npv          =  TN/(TN+FN)
    fall_out     = FP/(FP+TN)
    fnr          = FN/(FN+TP)
    fdr          = FP/FP+TP
    accuracy     = (TP+TN)/(TP+FN+TN+FP)
    f_score      = (2*TP)/(2*TP+FP+FN)
    mc           = (TP*TN - FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

        # prints the value
    print "\nsensitivity: ", sensitivity
    print "\n","specificity: ", specificity
    print "\nPrecision: ", Precision
    print "\nnegative predictive value: ",npv
    print "\nfall_out: ",fall_out
    print "\nfalse-negative_rate: ",fnr
    print "\nfalse_discovery_rate: ",fdr
    print "\naccuracy: ",accuracy
    print "\nF-score: ",f_score
    print "\nmathews correlation coefficient: ",mc



imagelist = []
edgemap_list = []

sigma = 1

imagelist.append('input_image.jpg')
imagelist.append('input_image2.jpg')
edgemap_list.append('output_image.png')
edgemap_list.append('output_image2.png')
for i in range(0,2):
    print "\nComputing Edge for Image ", i
    actual_image = canny(imagelist[i],sigma,13,5)
    im01 = Image.fromarray(actual_image)
    # image_name = 'outfile_'+str(i)
    # image_name += '.jpg'
    fig1 = plt.figure(i+1)
    fig1.suptitle('Edge using Canny Detector')
    plt.imshow(im01)


    im = Image.open(edgemap_list[i])
    ground_truth = np.array(im)

    #This block of code will make the image of only 2 levels all edge pixels are of intensity 255 and other pixels are of intensity 0
    #After making the image intensity levels as binary we go for computing the edge evaluation metrics
    for i in range(1,ground_truth.shape[0]):
        for j in range(1,ground_truth.shape[1]):
            if ground_truth[i][j]>0:
                ground_truth[i][j] = 255
    # plt.gray()
    # plt.figure(i+3)
    # plt.imshow(Image.fromarray(ground_truth))

    # for i in range(1,ground_truth.shape[0]):
    #     for j in range(1,ground_truth.shape[1]):
    #         print (ground_truth[i][j])
    plt.gray()
    fig2 = plt.figure(i+5)
    fig2.suptitle('ground Truth Edge Map')
    plt.imshow(Image.fromarray(ground_truth))
    print "\nComputing Qualitative Evaluation metrics for Image: ", i
    compute_metrics(ground_truth,actual_image)
    plt.show()





