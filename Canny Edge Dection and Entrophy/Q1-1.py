from __future__ import division
import Queue
from imaplib import _Authenticator

"""
    Below program will implement canny edge detector using below steps
    1) compuete 1D guassian  for X direction convolution and for y Direction convolution
    2) Convolves input Image with Guassian mask in x direction and we get I1
    3) Convolves input Image with Guassian mask in y direction ans we get I2
    4) Convolves I1 Image  Ix  with Guassian Derivative in x direction and we get Ix
    5) Convolves I2 Image  Iy  with Guassian Derivative in y direction and we get Iy
    6) Compute the magnitude Image M by combining the x and y components
    7) Computes non maxima suppression by taking the tangent and looking for neighbouring in the tangent direction
       and make that pixel to zero if  its neighbours are greater then this pixel
    8) Peforms hysterisis tresholding pixels

    Input Images used are:
    a)boat.jpg
    b)aeroplane.jpg
    c)pot.jpg
    Above Images are tested for sigma values 1,1.2,1.4,1.6 and among all 1.2 works better than other sigma values

"""

from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math


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


# def meanGauss(x,sigma):
#
#     value = (gaussian(x-0.5,sigma)+ gaussian(x,sigma)+gaussian(x+0.5,sigma))/3
#     value = value/(sigma* sigma*(math.pi*2))
#
#     return value

# def guassian_der(x,sigma):
#     return -x/(sigma*sigma) * gaussian(x, sigma);

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
    other better sigma values are 1.1 and 1.4
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
    plt.figure(1)
    plt.imshow(im01)


    im02 = Image.fromarray(Iy)
    image_list.append(im02)
    image_description.append('Image after smoothing in y direction')
    plt.figure(2)
    plt.imshow(im02)

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
    plt.figure(3)
    plt.imshow(im03)

    im04 = Image.fromarray(Iy1)
    image_list.append(im04)
    image_description.append('Image after applying guassian derivative in y direction')
    plt.figure(4)
    plt.imshow(im04)
    # #plt.show()

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
    plt.figure(5)
    plt.imshow(im05)

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
    plt.figure(7)
    plt.imshow(im07)
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
    for i in range(0,M1.shape[0]):
        for j in range(0,M1.shape[1]):
            #if abs(M1[i][j] -255.0) !=0 or abs(255-M1[i][j])!= 255:
            if M1[i][j] <= u_treshold:
                M1[i][j] = 0.0

    for i in range(0,M1.shape[0]):
        for j in range(0,M1.shape[1]):
            #if abs(M1[i][j] -255.0) !=0 or abs(255-M1[i][j])!= 255:
            if M1[i][j] - u_treshold > 0:
                M1[i][j] = 255.0


    im8 = Image.fromarray(M1)
    image_list.append(im8)
    image_description.append('Image after hysterisis')
    plt.figure(8)
    plt.imshow(im8)
    # plt.show()


    #plot each input image and binary image in single window

    # # generation of a dictionary of (title, images)
    # number_of_im = 6
    # figures = {image_description[i]:image_list[i]  for i in range(number_of_im)}

    # for i in range(0,len(image_list)):
    #     plt.figimage(i+1)
    #     plt.imshow(image_list[i])
    plt.show()



sigma = 1
# for high_t in range(4,30):
#     for low_t in range(1,high_t):
imagelist = []

imagelist.append('boat.jpg')
imagelist.append('pot.jpg')
imagelist.append('tiger.jpg')
for i in range(0,len(imagelist)):
    sigma = 1.0
    while sigma < 1.6:
        print 'current Image is :', i
        print '\ncurrent Sigma is :', sigma
        canny(imagelist[i],sigma, 20.0, 9.0)
        sigma += 0.2
        # print "low_t ", low_t, "  high_t", high_t
        # sigma += 0.1
print "Good egdes response is  detected for sigma value 1.2 among other sigma value"