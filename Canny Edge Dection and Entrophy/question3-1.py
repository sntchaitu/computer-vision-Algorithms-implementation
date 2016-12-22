from __future__ import division
import Queue
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import math
"""
below program computes the box blur on residual images for 30 iterations
and corespondingly a guassina filter is calculated for radius k = 1 to 15
and mse value of each iteration of box filter and image applied after each gauss filter of size 3*3 is compueted
and for after 1 iteration box fitered image equals to guass filter image of guass filter size  3*3
There are useful reference

width of box filter = math.sqrt((12*(sigma)*(sigma)/n)+1)equation 1 (reference http://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf)
and for a given sigma the guasian filter size need is

hence for box filter size of size 3*3 the corresponding no of recusive iteration to be done on box flter for getting appropriate guassian blur is for sigma = 1
9 = 12/n   + 1
12/n = 8
n = 4/3 = 1.33 approximately 1 iteration needed for guassian filter of size 3*3 and sigma value taken as 1


radius = round(sqrt(-log(T) * 2 * sigma^2)) equation 2              (reference http://suraj.lums.edu.pk/~cs436a02/CannyImplementation.htm)
hence when sigma = 1 appropriate guassian kernel size is 3X3



"""

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


def gen_gmask(radius,k,sigma):
    """
    computes the guassian window for particular window size
    """
    sum = 0
    g_window = np.zeros((1,k))
    for i in range(-radius,radius+1):
        den = 1/(sigma*math.sqrt(2*math.pi))
        g_window[0][i+radius] = den * gaussian(i,sigma)
        sum += g_window[0][i+radius]
    for i in range(-radius,radius+1):
        g_window[0][i+radius] = g_window[0][i+radius]/sum

    return g_window

def gaussian_blur(image,sigma,radius):
    """
    performs gaussian blur on image with kernel size k
    """
    width = 2*radius+1
    #compute 1 dimensional guassian kernel with size 1Xk
    gmask = gen_gmask(radius,width,sigma)

    #flip the mask
    i = 0
    j = len(gmask)-1
    while(i<j):
        temp = gmask[0][j]
        gmask[0][j] = gmask[0][i]
        gmask[0][i] = temp
        i+=1
        j-=1

    #pad the array with additional columns
    image_padded = np.pad(image, [(0, 0), (radius, radius)], 'edge')
    image_convolved_x = np.ones((image.shape[0], image.shape[1]), np.float32)


    # perform the convolution process along x direction

    for i in range(0, image.shape[0]):
        for j in range(radius, image.shape[1]+radius):
            sum1 = 0
            for n in range(-radius, radius+1):
                sum1 = sum1 + image_padded[i][j+n]*gmask[0][n+radius]
            image_convolved_x[i][j-radius] = sum1

    #transpose the mask
    gmask_y = gmask.transpose()


    #convolves image with the guassian mask transpose in Y direction
    image_padded2 = np.pad(image_convolved_x, [(radius, radius), (0, 0)], 'edge')
    image_convolved_y = np.ones((image.shape[0], image.shape[1]), np.float32)
    #Next convolves image  in Y direction
    # perform the convolution process along x direction

    # perform the convolution process along y direction



    for j in range(0, image.shape[1]):
        for i in range(radius, image.shape[0]+radius):
            sum1 = 0
            for n in range(-radius, radius+1):
                sum1 = sum1 + image_padded2[i+n][j]*gmask_y[n+radius][0]
            image_convolved_y[i-radius][j] = sum1

    return image_convolved_y


def boxblur(i1):
    """
    below function will compute box blur for image i1 first horizontally and next vertically
    """
    I1 = np.ones(i1.shape)

    I2 = np.ones(i1.shape)
    for i in range(0,I0.shape[0]):
        for j in range(0,I0.shape[1]):
            total = 0
            for k in range(-radius,radius+1):
                if(j+k>-1 and j+k<I0.shape[1]):
                    total= total + I0[i][j+k]
            I1[i,j] = total/(radius*2+1)


    for j in range(0,I1.shape[1]):
        for i in range(0,I1.shape[0]):
            total = 0
            for k in range(-radius,radius+1):
                if(i+k>-1 and i+k<I0.shape[0]):
                    total= total + I1[i+k][j]
            I2[i,j] = total/(radius*2+1)

    return I2

def compute_mse(image1,image2):
    """
    below function will compute mean square error which is the sum of the squared differences between two images .The lower the value the more similar
    """
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return  err

image_name = 'pot.jpg'
im = Image.open(image_name).convert('L')
I0 = np.array(im)
#print I0.shape
radius = 1

#plt.figure(1)
#plt.imshow(Image.fromarray(I0))

#perform gaussian blur on images
gblur_images = []
temp1 = np.empty_like(I0)
temp1[:] = I0
I_orig = np.empty_like(I0)
I_orig[:] = I0
for k in range(1,16):
    I0[:] = I_orig
    temp1 = gaussian_blur(I0,1,k)
    gblur_images.append(temp1)


for k in range(0,15):
    plt.gray()
    plt.figure(k+2)
    plt.imshow(Image.fromarray(gblur_images[k]))



boxblur_images = []
orig = np.empty_like(I0)
orig[:] = I0
temp = np.empty_like(I0)
temp[:] = I0
for k in range(0,30):
    temp = boxblur(I0)
    boxblur_images.append(temp[:])
    I0 = temp[:]
for k in range(0,30):
    plt.gray()
    plt.figure(17+k)
    plt.imshow(Image.fromarray(boxblur_images[k]))
min_value= 10000
iter = -1
k_final = 0
m_good = 0
n_good = 0
for  m in range(0,30):
    for n in range(0,15):
        t= compute_mse(boxblur_images[m],gblur_images[n])
        print t
        if(min_value > t):
            min_value = t
            m_good = m
            n_good = n
            iter = m+1
            k_final = 2*(n+1)+1

    plt.gray()
    plt.figure(1)
    plt.imshow(Image.fromarray(boxblur_images[m_good]))

    plt.figure(2)
    plt.imshow(Image.fromarray(boxblur_images[n_good]))
print "Box Blurred Image after ",iter,"iteration will more accurately match with image  that has been convolved with guassian image of kerel size and sigma 1",k_final, "*",k_final
print  iter,k_final


plt.show()
