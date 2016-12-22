from __future__ import division
from PIL import Image
import Queue
from pylab import *
import random
import matplotlib.pyplot as plt
import numpy as np
import math


"""
    Below program will compute create salt and pepper noise and guassian noise to the input image and compute
    apply edge detection algorithm on the image and finally compuete the below performance metrics on the ground truth
    image and actual computed edge image

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


def canny(I0,sigma,u_treshold,l_treshold):
    """
    computes the canny edges by ting sigma , u_treshold and l_treshold
    sigma value 1 the edge map is better than for other sigma values
    other better sigma values are 1.2 and 1.4
    :param sigma:
    :param u_treshold:
    :param l_treshold:
    :return:
    """
    image_list = []
    image_description = []
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


    for i in range(1,M.shape[0]):
        for j in range(1,M.shape[1]):
            if M1[i][j]<255:

                M1[i][j] = 0

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


############################################################################################################################
# metrics compution module starts
##############################################################################################################################
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

     #This block of code will make the image of only 2 levels all edge pixels are of intensity 255 and other pixels are of intensity 0
    #After making the image intensity levels as binary we go for computing the edge evaluation metrics
    for i in range(1,edge_map.shape[0]):
        for j in range(1,edge_map.shape[1]):
            if edge_map[i][j]>0:
                edge_map[i][j] = 255

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
#########################################################################################################
#metrics compution  module ends here
#########################################################################################################


#########################################################################################################
#below are the methods for adding guassian noise and salt and pepper noise
#########################################################################################################


def gen_snp_noise(img, std = 25):
    """
    below function will add salt and pepper noise to the image and return the image
    :param img_dim:
    :param std:
    :return:
    """

    noofpixels =  (std/100)* (img.shape[0]*img.shape[1])
    noofpixels = (int)(noofpixels)
    black_pixels = (int)(noofpixels/2)
    white_pixels = noofpixels-black_pixels
    #below array will keep track of what array index has changed in order to avoid duplicacy
    check_pixel = np.ones((img.shape[0],img.shape[1]), dtype = int)

    # below code block will add white noise to the image till the specified count is reached
    i = 0
    while i <white_pixels:
        rand_int_x = random.randint(0,img.shape[0]-1)
        rand_int_y = random.randint(0,img.shape[1]-1)
        if check_pixel[rand_int_x][rand_int_y] == 1:
            img[rand_int_x][rand_int_y] = 255
            check_pixel[rand_int_x][rand_int_y] = 2
            i  = i + 1
    # below code block will add black noise to the image till the specified count is reached
    j = 0
    while j < black_pixels:
        rand_int_x = random.randint(0,img.shape[0]-1)
        rand_int_y = random.randint(0,img.shape[1]-1)
        if check_pixel[rand_int_x][rand_int_y] == 1:
            img[rand_int_x][rand_int_y] = 0
            check_pixel[rand_int_x][rand_int_y] = 0
            j = j + 1
    return img



def add_guassian_noise(image, mean, std):
    """
    below function adds gaussian noise for all the pixels of the image
    :param image:
    :param mean:
    :param std:
    :return:Image
    """
    for i in range (0,image.shape[0]):
        for j in range (0,image.shape[1]):
            #genrates a gaussian noise with mean 0 and std = 1
            temp = image[i][j] + np.random.normal(0,10,1)
            if temp < 0:
                image[i][j] = 0;
            elif temp > 255:
                image[i][j] = 255;
            else:
                image[i][j] = temp;
    return image


###################################################################################################
# Driver Program to test the above functions
######################################################################################################


# loads the input image from directory
im_in = Image.open('input_image.jpg').convert('L')
im_out = Image.open('output_image.png').convert('L')
edgemap_array = np.array(im_out)
im1 = np.array(im_in)
im01 = Image.fromarray(im1)
plt.gray()
plt.figure(1)
plt.imshow(im01)
#adds gaussian noise to the image
im_g_added = add_guassian_noise(im1, 0,10)
im02 = Image.fromarray(im_g_added)
plt.figure(2)
plt.suptitle('Image after gaussian noise')
plt.imshow(im02)
#im_g_added = im1

#adds salt and pepper noise to the image
im_snp_added = gen_snp_noise(im_g_added,25)

im03 = Image.fromarray(im_snp_added)

###########################################################
# save im03 image
########################################################

plt.figure(3)
plt.suptitle('Image after salt and pepper noise')
plt.imshow(im03)
#plt.show()
im_snp_added = np.array(im_in)
#computes canny edge with given parameters
sigma = 1
u_tresh = 14
l_tresh = 5
computed_edge = canny(im_snp_added,sigma,u_tresh,l_tresh)
plt.suptitle('Image after canny edge detection')
im04 = Image.fromarray(computed_edge)
plt.figure(4)
plt.imshow(im04)


# Computing Qualitative Evaluation metrics of edge detector for given image
compute_metrics(edgemap_array,computed_edge)
plt.show()