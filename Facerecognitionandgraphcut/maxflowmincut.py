from PIL import Image
from scipy import ndimage
from scipy import signal
import numpy as np
import scipy
import cv2
import glob
import maxflow
import matplotlib.pyplot as plt
import math
i1 = Image.open('aero.jpg').convert('L')
i2 = cv2.imread('aero.jpg',0)
I1 = np.array(i1)
#below program will run the graph cut segmentation for user defined fore ground and back ground
#uses is reqired to crop a region on the iimage using mouse clicks
#press key c to close the mouse action
#Now probabilitty that each pixel as backgroun and foreground is calcualted using notmalized histogram probabilties
#Energy is given by the equation
#Rp is the region penalities for fore gorund and back goeund
#E(A) = ld*Rp+ B
#B is the cost factor between pixel(Pa,Pb) where pa and pb does not belong to fore ground and back ground
#http://pmneila.github.io/PyMaxflow/tutorial.html#binary-im_original-restoration
#now we will use the maxflow() function of pymaxflow library which will sever the graph into 2 labels fore ground and background
#display the final im_original afer the segmentation
#reference eqn for unary potantial
#https://courses.engr.illinois.edu/cs543/sp2011/lectures/Lecture%2012%20-%20MRFs%20and%20Graph%20Cut%20Segmentation%20-%20Vision_Spring2011.pdf

#click on mosse and drag a rectangle and press key c to  extract the window of region of interest

print 'select 2 points on the image and it is treated as foreground'

plt.figure(1)
plt.imshow(i1)
points = plt.ginput(2)

plt.show()

x1,y1 = points[0]
x2,y2 = points[1]
x1,y1 = int(x1),int(y1)
x2,y2 = int(x2),int(y2)

#select the background region
#Computing Histograms to set regional penalties
background_mask = np.ones(I1.shape,np.uint8)
background_mask[x1:x2,y1:y2] = 0
background_histogram = cv2.calcHist([i2],[0],background_mask,[256],[0,256])
foreground_mask = np.zeros(I1.shape,np.uint8)
foreground_mask[x1:x2,y1:y2] = 255
foreground_histogram = cv2.calcHist([i2],[0],foreground_mask,[256],[0,256])
#normalize the histograms to compute probability of foreground and background
foreground_histogram /= np.max(foreground_histogram)
background_histogram /= np.max(background_histogram)

#background intensity map
plt.figure(2)
plt.title('backgorund intensity map')
plt.plot(background_histogram)#foregroundintensity map
plt.figure(3)
plt.title('foreground intensity map')
plt.plot(foreground_histogram)

#compuete regional penalities
image_vectorized = I1.ravel()
penalty_background = [-math.log(background_histogram[y1]+1e-6) for y1 in image_vectorized]
penalty_foreground = [-math.log(foreground_histogram[y1]+1e-6) for y1 in image_vectorized]

# #Compute  penalties for the energy function for background region and foreground region
Regional_penalty = [-math.log((foreground_histogram[x]+1e-6)/(foreground_histogram[x]+1e-6)) for x in image_vectorized]
energy_unary = np.reshape(Regional_penalty,(I1.shape[0],I1.shape[1]))

# #calcualting pair wise energy potentials B for the entire im_original
# #using the equation B = exp power (-(Ip- Iq)*(Ip- Iq)/2)
# #here for each pixel intensity value 8 neighbourhood intensity values are considered.

pixel_neighbours = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
B = np.zeros(I1.shape)
vectorized_image = B.reshape(I1.shape[0]*I1.shape[1])
padded_image = np.pad(I1,(1,),'reflect').astype('float64')
for y in range(1,I1.shape[0]) :
    for x in range(1,I1.shape[1]) :
        neighbors = [[p[0]+y,p[1]+x] for p in pixel_neighbours]
    pairwise_sum = 0
    for i in range(len(pixel_neighbours)) :
        p = [neighbors[i][0],neighbors[i][1]]
        pairwise_sum += np.exp(-((padded_image[y,x]-padded_image[p[0],p[1]])**2)/2)
    indx = (y-1)*I1.shape[1]+(x-1)
    vectorized_image[indx] = pairwise_sum
energy_pairwise = np.reshape(vectorized_image,(I1.shape[0],I1.shape[1]))

# #Energy function
# totalenergy = lambda*Rp + B
# # where B  is pair wise energy functionals
# #lambda*Rp is unary potentials
#compuete maxflow using pymaxflow inbuilt function
ld = 27
total_energy = ld*energy_unary + energy_pairwise


G = maxflow.Graph[int](0,0)
nodeids = G.add_grid_nodes(I1.shape)
G.add_grid_edges(nodeids, weights=total_energy)
G.add_grid_tedges(nodeids, I1, 255-I1)
G.maxflow()
results = G.get_grid_segments(nodeids)
# The labels should be 1 where sgm is False and 0 otherwise.
final_image = np.invert(results)
plt.figure(4)
plt.title('final segmented image')
plt.imshow(final_image)
plt.show()




