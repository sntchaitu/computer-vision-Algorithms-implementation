from __future__ import division
import Queue
from imaplib import _Authenticator

from sys import  argv
"""
    Below program will implement sift algorithm
    It has following stages
    1)guassian scale-space generation and extrema detection
    2)Key point localization
    3)orientation assignment
    4)Key point descriptor


    1)guassian scale-space generation and extrema detection


      for generating guassian scale space we assume 4 octave level and each octave has 5 levels of spaces
      Intially on the input image is  guassian blur is applied with initialblur of sigma value 0.5nad  upsampled by size facotr of 2 .
      Now for this input image we generate 5 levels of guassian blur (all 5 set of images are of same size and has same sampling distance between 2 pixels)
      but dffer in blur levels with factor of sqrt(2).

      for first octave
        blur levels are as follows 1/sqrt(2) 1  sqrt(2) 2  2sqrt(2)

      Now for the second octave we take middle image from the previous scale space
        and proceed with same blurring
        hence
        blur levels as sqrt(2)    2       2sqrt(2)  4    4sqrt(2)


        and we continue for the rest of 2 octaves  in similar way

    Now we compuete difference of guassian for each octave
    DOG(1) = Scale space Image (2) - Scale space Image (1)
    DOG(2) = Scale space Image (3) - Scale space Image (2)
    DOG(3) = Scale space Image (4) - Scale space Image (3)
    DOG(4) = Scale space Image (5) - Scale space Image (4)


    After obtaining all the scale space we check for maximum pixel among the neighbour pixels as well as in neighbouring scale space
    hence we obtain 2 extrema images for each octave

    Now the key points are further refined using key point localization

    Each keypoint is assigned one or more orientations based on local image gradient directions.

    128 length sift descriptor is calculated

    sift key points are plotted on both input images



"""

from PIL import Image
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
from scipy import ndimage
import cv2


def sift(image_name,number):
    sigma = 1.6
    k = math.pow(2,1/2)
    scales = 5;
    octaves = 4;

    I = Image.open(image_name).convert('L')
    i0 = np.array(I)
    #upsample the original image
    #upsampled = np.zeros((i0.shape[0]*2,i0.shape[1]*2))
    #upsampled[:] = cv2.resize(i0,(i0.shape[1]*2,i0.shape[0]*2))
    #I_original = np.zeros(upsampled.shape)
    #I_original[:] = upsampled
    I_original = np.zeros(i0.shape)
    I_original[:] = i0

    octave_list = []
    rows = I_original.shape[0]
    cols = I_original.shape[1]
    print "rows is",rows
    print "colss is",cols




    """
        first get all the base images of all octaves and store in the list
    """
    base_list = []
    for i in range(0,octaves):
        scale_list = []
        for j in range(0,scales):
             if i==0 and j ==0:
                 scale_list.append(cp.deepcopy(I_original))
             elif i>0 and j==0:
                 temp = cp.deepcopy(ndimage.zoom(base_list[i-1][0],0.5,order=1))
                 scale_list.append(temp)

        base_list.append(scale_list)

    # ######################################print the base list images##################################################
    # count = 40;
    # for i in range(0,len(base_list)):
    #     #for j in range(0,len(octave_list[0])):
    #     plt.figure(count)
    #     plt.imshow(Image.fromarray(base_list[i][0]))
    #     count+=1
    # #plt.show()
    # ######################################################################################################################


    ####################################################below module will create octave list with 4 octaves and 5 scales for each octave########################################

    for i in range(0,octaves):
        scale_list = []
        # if i ==0:
        #     I_base = np.zeros(i0.shape)
        #     I_base[:] = i0

        for j in range(0,scales):
            if j ==0:
                temp= np.zeros(base_list[i][0].shape)
                temp[:] = base_list[i][0]

            sigma = math.pow(k,j)*1.6
            hsize = int(math.ceil(7*sigma))
            hsize = 2*hsize+1


            i_temp = np.zeros(temp.shape)
            i_temp1 = np.zeros(temp.shape)

            i_temp[:] = cv2.GaussianBlur(temp,(hsize,hsize),sigma,sigma)

            scale_list.append(i_temp)

        #make a copy of base imahe before resizeing
        # I_copy = np.zeros(I_base.shape)
        # I_copy[:] = I_base
        # if(i<3):
            # #resize the image
            # rows = I_copy.shape[0]/2
            # cols = I_copy.shape[1]/2
            # if rows-int(rows)!=0.0:
            #     rows = int(rows+1)
            # else:
            #     rows = int(rows)
            # if cols-int(cols)!=0.0:
            #     cols = int(cols+1)
            # else:
            #     cols = int(cols)
            #
            # print "rows is",rows
            # print "colss is",cols

            # temp = ndimage.zoom(I_copy,0.5,order=1)
            # I_base = np.zeros(temp.shape)
            # I_base[:] = temp
        octave_list.append(scale_list)

    ####################################################octave list creation ends here ########################################

    # # ######################################print the octave list images##################################################
    # count = 1;
    # for i in range(0,len(octave_list)):
    #     for j in range(0,len(octave_list[0])):
    #         plt.figure(count)
    #         plt.imshow(Image.fromarray(octave_list[i][j]))
    #         count+=1
    # plt.show()
    # # ######################################################################################################################

    ####################################compute DOG LIST###########################################################
    DOG_LIST = []
    for i in range(0,octaves):
        scale_list = []
        for j in range(1,scales):
            #Get the temporal different It betwenn 2 guassian images
            diff = np.zeros(octave_list[i][0].shape)
            diff[:] =  np.subtract(octave_list[i][j],octave_list[i][j-1])
            #diff[diff<0] = 0
            scale_list.append(diff)
        DOG_LIST.append(scale_list)
    ##############################################################DOG LIST ENDS HERE###############################

     ######################################print the DOG list images##################################################
    # count = 1;
    # # print len(DOG_LIST)
    # # print len(DOG_LIST[0])
    # for i in range(0,len(DOG_LIST)):
    #     for j in range(0,len(DOG_LIST[0])):
    #         plt.figure(count)
    #         plt.imshow(Image.fromarray(DOG_LIST[i][j]))
    #         count+=1
    # #plt.show()
    # ######################################################################################################################
    count1 = 0;
    ############################################################Compuete Extrema set of Images###########################
    Extrema_List = []
    for i in range(0,octaves):
        scale_list = []
        for j in range(1,scales-2):
            ext_image = np.zeros(DOG_LIST[i][j].shape,dtype=np.float64)
            for m in range(1,DOG_LIST[i][j].shape[0]):
               for n in range(1,DOG_LIST[i][j].shape[1]):
                   value = DOG_LIST[i][j][m][n]
                   if(value== max(DOG_LIST[i][j][m-1:m+2,n-1:n+2].max(),DOG_LIST[i][j-1][m-1:m+2,n-1:n+2].max(),DOG_LIST[i][j+1][m-1:m+2,n-1:n+2].max())):
                       ext_image[m][n] = value
                       count1=count1+1;
                   elif(value== min(DOG_LIST[i][j][m-1:m+2,n-1:n+2].min(),DOG_LIST[i][j-1][m-1:m+2,n-1:n+2].min(),DOG_LIST[i][j+1][m-1:m+2,n-1:n+2].min())):
                       ext_image[m][n] = value
                       count1= count1+1
            scale_list.append(ext_image)
        Extrema_List.append(scale_list)
    print "extream count is",count1
    ############################################################Compuete Extrema set of Images ends here###########################

    ######################################print EXtrema list images##################################################
    # count = 1;
    # # print len(DOG_LIST)
    # # print len(DOG_LIST[0])
    # for i in range(0,len(Extrema_List)):
    #     for j in range(0,len(Extrema_List[0])):
    #         plt.figure(count)
    #         plt.imshow(Image.fromarray(Extrema_List[i][j]))
    #         count+=1
    # plt.show()
    ######################################################################################################################

    ##########################append the list of non zero elements of the extrema list#######################################################
    no_of_keypoints = 0
    non_zero_sigma = []
    non_zero_extrema = []
    for i in range(0,octaves):
        scale_list = []
        sigma_list = []
        for j in range(0,scales-3):
            temp_list = []
            temp_list[:] = np.transpose(Extrema_List[i][j].nonzero())
            no_of_keypoints+=len(temp_list)
            scale_list.append(temp_list)
            sigma_list.append(math.pow(k,j)*1.6)
        non_zero_extrema.append(scale_list)
        non_zero_sigma.append(sigma_list)

    print(no_of_keypoints)


    # ##################################################plot all key points before Loc###################################################
    c = 3

    # plt.gray()
    # plt.title('key points before localization');
    # plt.figure(2*number+1)
    #
    #
    # plt.imshow(I)

    for i in range(0,octaves):
        for j in range(0,2):
            #print "i = ",i,"j = ",j,len(non_zero_extrema[i][j])
            for p in range(0,len(non_zero_extrema[i][j])):
                #get the new x,y co-ordinates with respect to that octave scale
                x1 = math.pow(2,i)*non_zero_extrema[i][j][p][0]
                y1 = math.pow(2,i)*non_zero_extrema[i][j][p][1]


                # dx =  c*extPts2[i][j][p][3] * math.degrees(math.cos(extPts2[i][j][p][10]))
                # dy =  c*extPts2[i][j][p][3] * math.degrees(math.sin(extPts2[i][j][p][10]))
                t1 = [x1]
                t2 = [y1]
                #plt.plot(t2,t1,'r*')
                #plt the arrow
                #plt.arrow(x1,y1,dx,dy,color = 'yellow',head_width = 6,head_length = 7)
    ########################################################################################################################


    #####################################################################################################################
    # print non_zero_extrema[0][0][0]
    # for i in range(0,octaves):
    #     for j in range(0,2):
    #         print(non_zero_extrema[i][j])
    ###################################################################################################################

    #########################################Key Point Localization part###############################################
    keyPtCounter = 1;
    extPts2 = []
    counter2 = 0
    for i in range(0,octaves):
        scale_list = []
        for j in range(0,2):
            keyPtCounter = 1
            kpperscale = []
            #print "i=",i,"j=",j,len(non_zero_extrema[i][j])
            for m in range(0,len(non_zero_extrema[i][j])):
                matA = np.zeros((3,3))
                matB = np.zeros((3,1))

                xPt = non_zero_extrema[i][j][m][0]
                yPt = non_zero_extrema[i][j][m][1]
                cur_sigma = non_zero_sigma[i][j]
                if(xPt+1<DOG_LIST[i][0].shape[0] and yPt+1<DOG_LIST[i][0].shape[1]  and xPt-1>-1 and yPt-1>-1):
                    xHatNew  = np.zeros((3,1))
                    xPtNew = xPt
                    yPtNew = yPt
                    sigNew = cur_sigma


                # matA(1,1) = IDoG{j+1-1,i}(yPt,xPt) - 2*IDoG{j+1,i}(yPt,xPt) + IDoG{j+1+1,i}(yPt,xPt);
                # matA(1,2) = IDoG{j+1+1,i}(yPt+1,xPt) - IDoG{j+1+1,i}(yPt-1,xPt) - IDoG{j+1-1,i}(yPt+1,xPt) + IDoG{j+1-1,i}(yPt-1,xPt);
                # matA(1,3) = IDoG{j+1+1,i}(yPt,xPt+1) - IDoG{j+1+1,i}(yPt,xPt-1) - IDoG{j+1-1,i}(yPt,xPt+1) + IDoG{j+1-1,i}(yPt,xPt-1);
                # matA(2,1) = matA(1,2);
                # matA(2,2) = IDoG{j+1,i}(yPt+1,xPt) - 2*IDoG{j+1,i}(yPt,xPt) + IDoG{j+1,i}(yPt-1,xPt);
                # matA(2,3) = IDoG{j+1,i}(yPt-1,xPt-1) - IDoG{j+1,i}(yPt+1,xPt-1) - IDoG{j+1,i}(yPt-1,xPt+1) + IDoG{j+1,i}(yPt+1,xPt+1);
                # matA(3,1) = matA(1,3);
                # matA(3,2) = matA(2,3);
                # matA(3,3) = IDoG{j+1,i}(yPt,xPt+1) - 2*IDoG{j+1,i}(yPt,xPt) + IDoG{j+1,i}(yPt,xPt-1);
                #
                # matB(1,1) = IDoG{j+1+1,i}(yPt,xPt) - IDoG{j+1-1,i}(yPt,xPt);
                # matB(2,1) = IDoG{j+1,i}(yPt+1,xPt) - IDoG{j+1,i}(yPt-1,xPt);
                # matB(3,1) = IDoG{j+1,i}(yPt,xPt+1) - IDoG{j+1,i}(yPt,xPt-1);
                #
                # %xHat = inv(matA) * matB;
                # xHat = matA\matB;

                # xHatNew = xHat;
                #
                # skipPt = 0;
                    matA[0][0] = DOG_LIST[i][j+1-1][xPt][yPt] - 2*DOG_LIST[i][j+1][xPt][yPt] + DOG_LIST[i][j+1+1][xPt][yPt]
                    matA[0][1] = DOG_LIST[i][j+1+1][xPt+1][yPt] -DOG_LIST[i][j+1+1][xPt-1][yPt] - DOG_LIST[i][j-1+1][xPt+1][yPt] + DOG_LIST[i][j-1+1][xPt-1][yPt];
                    #if(keyPtCounter==1008):
                        #print " here1"
                    matA[0][2] = DOG_LIST[i][j+1+1][xPt][yPt+1] -DOG_LIST[i][j+1+1][xPt][yPt-1] - DOG_LIST[i][j-1+1][xPt][yPt+1] + DOG_LIST[i][j-1-1][xPt][yPt-1];
                    matA[1][0] = matA[0][2]
                    matA[1][1] = DOG_LIST[i][j+1][xPt+1][yPt] - 2*DOG_LIST[i][j+1][xPt][yPt] + DOG_LIST[i][j+1][xPt-1][yPt]

                    matA[1][2] = DOG_LIST[i][j+1][xPt-1][yPt-1] - DOG_LIST[i][j+1][xPt+1][yPt-1]  - DOG_LIST[i][j+1][xPt-1][yPt+1] + DOG_LIST[i][j+1][xPt+1][yPt+1]
                    matA[2][0] = matA[0][2]
                    matA[2][1] = matA[1][2]
                    matA[2][2] = DOG_LIST[i][j+1][xPt][yPt+1] - 2*DOG_LIST[i][j+1][xPt][yPt] + DOG_LIST[i][j+1][xPt][yPt-1]

                    matB[0][0] =  DOG_LIST[i][j+1+1][xPt][yPt] - DOG_LIST[i][j+1-1][xPt][yPt]
                    matB[1][0] =  DOG_LIST[i][j+1][xPt+1][yPt]- DOG_LIST[i][j+1][xPt-1][yPt]
                    matB[2][0] =  DOG_LIST[i][j+1][xPt][yPt+1]- DOG_LIST[i][j+1][xPt][yPt-1]

                    #Hat = inv(matA) * matB;
                    xHat = np.dot(np.linalg.pinv(matA),matB)

                    xHatNew[:] = xHat
                    skipPt = 0
                    # if(keyPtCounter<2):
                    #     print xHat[0],xHat[1],xHat[2   ]
                    #     print xHat[0][0],xHat[1][0],xHat[2][0]
                    #     print xHat.shape

                    # print("keypointounter",keyPtCounter)
                    # print("octave is",i)
                    # print "scale is ",j


                    ############################## Change Pts having offset greater than 0.5 in any dimension###################################################
                    if((abs(xHat[0][0])>0.5) or (abs(xHat[1][0])>0.5) or (abs(xHat[2][0])>0.5)):
                        skipPt = 1
                        if abs(xHat[1][0])>0.5:
                            xPtNew = xPt + round(xHat[1][0])
                            xHatNew[1][0] = xHat[1][0]-round(xHat[1][0]);
                            if (xPtNew>octave_list[i][0].shape[0]-1) or (xPtNew<0):
                                skipPt = 1



                        if abs(xHat[2][0])>0.5:
                            yPtNew = yPt + round(xHat[2][0])
                            xHatNew[2][0] = xHat[2][0]-round(xHat[2][0])

                            if (yPtNew>octave_list[i][0].shape[1]-1) or (yPtNew<0):
                                skipPt = 1;


                        if abs(xHat[0][0])>0.5:
                            if xHat[0][0]>0:
                                sigNew = math.pow(k,(j+1))*1.6
                                xHatNew[0][0] = (sigNew - math.pow(k,(j))*1.6) - xHat[0][0]
                            else:
                                sigNew = math.pow(k,(j-1))*1.6
                                xHatNew[0][0] = (math.pow(k,j)*1.6 - sigNew) + xHat[0][0]


                    #########################################check for low contrast key points##################################################################
                    if (skipPt == 0):
                        contrast = DOG_LIST[i][j+1][xPtNew][yPtNew] + 0.5 *(matB[1][0]*xHatNew[2][0]+ matB[2][0]*xHatNew[2][0] + matB[0][0]*xHatNew[0][0])

                        # Do Nothing

                        if(abs(contrast)>0.03):
                        ###################################### check for poor edge localizations###############################################
                            Dxx = DOG_LIST[i][j+1][xPt+1][yPt] - 2*DOG_LIST[i][j+1][xPt][yPt] + DOG_LIST[i][j+1][xPt-1][yPt]
                            Dxy = DOG_LIST[i][j+1][xPt-1][yPt-1] - DOG_LIST[i][j+1][xPt+1][yPt-1] + DOG_LIST[i][j+1][xPt-1][yPt+1] +DOG_LIST[i][j+1][xPt+1][yPt+1]
                            #Dxy = IDoG{j+1,i}(yPt-1,xPt-1) - IDoG{j+1,i}(yPt+1,xPt-1) - IDoG{j+1,i}(yPt-1,xPt+1) + IDoG{j+1,i}(yPt+1,xPt+1);
                            Dyy = DOG_LIST[i][j+1][xPt][yPt+1] - 2*DOG_LIST[i][j+1][xPt][yPt] + DOG_LIST[i][j+1][xPt][yPt-1]
                            #Dyy = IDoG{j+1,i}(yPt+1,xPt) - 2*IDoG{j+1,i}(yPt,xPt) + IDoG{j+1,i}(yPt-1,xPt);
                            trH = Dxx + Dyy
                            detH = Dxx*Dyy - Dxy*Dxy
                            curvature_ratio = (trH*trH)/detH
                            if(abs(curvature_ratio)<10.0):
                                keypt_attributes = []
                                # Hess = [[Dxx, Dxy], [Dxy, Dyy]]
                                # eigVals = eig(Hess);
                                # r = max(eigVals)/min(eigVals);
                                #
                                # if (r<=10)


                                # % Save the KeyPt
                                # % 0 - keypointcounter
                                # % 1 - new x value
                                # % 2 - new y value
                                # % 3 - new sigma value
                                # % 4 - x offset
                                # % 5 - y offset
                                # % 6 - sigma offset
                                # % 7 - X value
                                # % 8 - Y value
                                # % 9 - sigma value
                                # % 10-scale value
                                # % 11- max theta
                                # % 12 max magnitude
                                keypt_attributes.append(keyPtCounter)
                                keypt_attributes.append(xPtNew)
                                keypt_attributes.append(yPtNew)
                                keypt_attributes.append(sigNew)
                                # extPts2{j,i}(keyPtCounter,1) = xPtNew;
                                # extPts2{j,i}(keyPtCounter,2) = yPtNew;
                                # extPts2{j,i}(keyPtCounter,3) = sigNew;
                                keypt_attributes.append(xHatNew[1][0])
                                keypt_attributes.append(xHatNew[2][0])
                                keypt_attributes.append(xHatNew[0][0])
                                keypt_attributes.append(xPt)
                                keypt_attributes.append(yPt)
                                keypt_attributes.append(cur_sigma)
                                keypt_attributes.append(j+1)
                                # extPts2{j,i}(keyPtCounter,4) = xHatNew(3,1);
                                # extPts2{j,i}(keyPtCounter,5) = xHatNew(2,1);
                                # extPts2{j,i}(keyPtCounter,6) = xHatNew(1,1);
                                # extPts2{j,i}(keyPtCounter,7) = xPt;
                                # extPts2{j,i}(keyPtCounter,8) = yPt;
                                # extPts2{j,i}(keyPtCounter,9) = sig;
                                keyPtCounter = keyPtCounter + 1;
                                kpperscale.append(keypt_attributes)
                                counter2 +=1
                                #print "Kp att is",keypt_attributes
            scale_list.append(kpperscale)
        extPts2.append(scale_list)

    # ##################################################plot all key points###################################################
    c = 3
    print "kp after local",counter2
    #plt.gray()
    #plt.title('key points after localization');
    #plt.figure(2*number+2)
    #plt.imshow(I)
    #plt.show()
    for i in range(0,octaves):
        for j in range(0,2):
            #print "i = ",i,"j = ",j,len(extPts2[i][j])
            for p in range(0,len(extPts2[i][j])):
                #get the new x,y co-ordinates with respect to that octave scale
                x1 = math.pow(2,i)*extPts2[i][j][p][1]
                y1 = math.pow(2,i)*extPts2[i][j][p][2]


                # dx =  c*extPts2[i][j][p][3] * math.degrees(math.cos(extPts2[i][j][p][10]))
                # dy =  c*extPts2[i][j][p][3] * math.degrees(math.sin(extPts2[i][j][p][10]))
                t1 = [x1]
                t2 = [y1]
                #plt.plot(t2,t1,'r*')
                #plt the arrow
                #plt.arrow(x1,y1,dx,dy,color = 'yellow',head_width = 6,head_length = 7)
    ########################################################################################################################

    ###############################calculate orientation of the key points##################################
    extPts3 = []
    count5 = 0
    for i in  range (0,octaves):
        scale_list = []
        for j in range(0,(scales-3)):
            keyPtCounter = 1
            #print "i ",i,"j ",j
            #print "len(extPts2[i][j])",len(extPts2[i][j])
            kpperscale = []
            for p in  range(0,len(extPts2[i][j])):
                #print 'cur kp is',p
                #print 'cur kp attributes',extPts2[i][j][p]
                #keypt_attributes = []
                xPt = extPts2[i][j][p][1]
                yPt = extPts2[i][j][p][2]
                sig = extPts2[i][j][p][3]
                IOr = np.zeros(octave_list[i][j].shape)
                IOr = octave_list[i][j];
                hsize = int(math.ceil(7*sig))
                #print hsize
                #print sig
                #H = cv2.getGaussianKernel(hsize,sig)
                Iblur = np.zeros(IOr.shape)
                #i_temp[:] = cv2.filter2D(temp,-1,H)
                H = cv2.getGaussianKernel(hsize,int(sig));
                Iblur[:                                                                                                                                                                                                                                                                                                                                                                                                                                                     ] = cv2.filter2D(IOr,-1,H);

                bins = np.zeros((1,36));

                for s in range(-hsize,hsize+1):
                    for t in range(-hsize,hsize+1):
                        if (((xPt + s)>0) and ((xPt + s)<(Iblur.shape[0]-1)) and ((yPt + t)>0) and ((yPt + t)<(Iblur.shape[1]-1))):
                            xmag1 = Iblur[xPt+s+1][yPt+t]
                            xmag2 = Iblur[xPt+s-1][yPt+t]
                            ymag1 = Iblur[xPt+s][yPt+t+1]
                            ymag2 = Iblur[xPt+s][yPt+t-1]
                            m = math.sqrt(math.pow((xmag1-xmag2),2)+math.pow((ymag1-ymag2),2))
                            #m = math.sqrt((Iblur[xPt+s][yPt+t]-Iblur[xPt+sd-1][(yPt+t])) + (Iblur(yPt+t+1,xPt+s)-Iblur(yPt+t-1,xPt+s))^2);
                            #theta = atand((Iblur(yPt+t+1,xPt+s)-Iblur(yPt+t-1,xPt+s))/(Iblur(yPt+t,xPt+s+1)-Iblur(yPt+t,xPt+s-1)));
                            den = xmag2-xmag1
                            if den==0:
                               den = 5
                            theta = math.degrees(math.atan((ymag2-ymag1)/(den)))
                            if(theta<0):
                                theta = 360 + theta

                            #binNum = (math.floor(theta/10) + 1)
                                #int bin = (int)(ori/(2*M_PI)*nbins+0.5)%nbins;
                            binNum = (int)((theta/360)*36)%36
                            #print theta
                            #print binNum
                            if binNum ==36:
                                binNum = 35
                            bins[0][binNum] = bins[0][binNum] + m

                maxBinNo = np.argmax(bins)
                maxtheta = maxBinNo*10
                maxmag = bins[0][maxBinNo]
                extPts2[i][j][p].append(maxtheta)
                extPts2[i][j][p].append(maxmag)

#####################################################find principle orientations###############################################
                nbins = 36
                threshold = 0.8
                principal_orientations = []
                o = 0
                for y in range(0,36):
                    orientation = 0
                # for(int i = 0; i < nbins; i++)
                    y_prev = (y-1+nbins)%nbins;
                    y_next = (y+1)%nbins;
                    if bins[0][y] > threshold*maxtheta and bins[0][y] > bins[0][y_prev] and  bins[0][y]> bins[0][y_next]:
                    #Quadratic interpolation of the position of each local maximum
                        #offset = interpolate_peak(bins[i_prev], bins[i], bins[i_next])
                        offset = (bins[0][y_prev] - bins[0][y_next])/(2*(bins[0][y_prev]+bins[0][y_next]-2*bins[0][y]))

                    #get the orientation corresponding to bin number
                        exact_bin = y+offset
                        orientation = exact_bin*360/float(36)
                        if orientation>360:
                            orientation-=360
                    #Add to vector of principal orientations
                        # principal_orientations.append(bin_to_ori((float)y + offset, nbins))
                        # float ori = (bin+0.5)*2*M_PI/(float)nbins;
                        # if (ori > M_PI)
                        # ori -= 2*M_PI;
                        # return ori;
                        o+=1
                    #now for each principle orientation create a key point attribute that is all parameters will be same for 2 keypoints and they differ only in principle orientation
                        keypt_attributes = []
                        keypt_attributes[:] = extPts2[i][j][p]
                        keypt_attributes[11] = orientation
                        kpperscale.append(keypt_attributes)
            count5 +=len(kpperscale)
            scale_list.append(kpperscale)
        extPts3.append(scale_list)
    print "after principle orientation",count5
                # extPtsGradTheta{j,i}(p,1) = (maxY*10) - 5;
                # extPtsGradTheta{j,i}(p,2) = m;
    #######################################################################################################################
# static float interpolate_peak(float h1, float h2, float h3)
# {
#     float offset = (h1-h3)/(2*(h1+h3-2*h2));
#     return offset;
# }

#######################compute gradient for scale space###########################################################
    dx_list  = []
    dy_list  = []
    for i in range(0,len(octave_list)):
        scale_list1 = []
        scale_list2 = []
        for j in range(0,scales):
            dx,dy = np.gradient(octave_list[i][j])
            scale_list1.append(dx)
            scale_list2.append(dy)
        dx_list.append(scale_list1)
        dy_list.append(scale_list2)

##################################################################################################################

# ##############################################################frame the feature descriptor of length 128 for each key point################################################################
# # #for each keypoint present in the list frame the 128 length feature descriptor
#     no_of_histograms = 16
#     no_of_bins = 8
#     n_hist = 4
#     m1 = 0
#     for i in range(0,len(extPts3)):
#         for j in range(0,2):
#             for k in range(0,len(extPts3[i][j])):
#                 # Initialize 128 width descriptor for each key point
#                 desc = np.zeros(no_of_bins*no_of_histograms)
#                 # if m1 ==0:
#                 #     print extPts3[i][j]
#                 #     print extPts3[i][j][1]
#                 #     print extPts3[i][j][1][1]
#                 #     m1 = 1
#                 x_key_pos = extPts3[i][j][k][1]
#                 y_key_pos = extPts3[i][j][k][2]
#                 sigma_key = extPts3[i][j][k][3]
#                 theta_key = extPts3[i][j][k][11]
#                 #for each key point frame 16*16 boundary around it and for each pixel in it compute the effective rotation of each pixel
#                 #and
#                 siMin = max(0, (int)(x_key_pos -8))
#                 sjMin = max(0, (int)(y_key_pos -8))
#                 siMax = max((int)(x_key_pos + 8), dx_list[i][j+1].shape[0]-1);
#                 sjMax = max((int)(y_key_pos + 8), dy_list[i][j+1].shape[1]-1);
#
#                 for a1 in range(siMin,siMax+1):
#                     for b1 in range(sjMin,sjMax+1):
#                         X = a1 - x_key_pos
#                         Y = b1 - y_key_pos
#
#                         #rotate the x and y locations towards key point reference orientation
#                         c = math.cos(-theta)
#                         s = math.sin(-theta)
#                         tx = c * X - s * Y
#                         ty = s * X + c * Y
#                         X = tx
#                         Y = ty
#
#                         if(siMin<=X and X<= siMax and sjMin<=Y and Y<= sjMax):
#                             #Compute the gradient orientation on keypoint referential.
#                             dy = dx_list[i][j][X][Y]
#                             dx = dy_list[i][j][X][Y]
#                             ori = math.degrees(math.atan2(dy, dx)) - theta_key
#                             if(ori<0):
#                                 ori = 360 + ori
#
#
#                           # Compute the gradient magnitude and apply a Gaussian weighing to give less emphasis to distant sample
#                             t = 1.5*sigma_key
#                             M = math.hypot(dx,dy) * math.exp(-(X*X+Y*Y)/(2*t*t))
#
#                             # compute linear weightings across histograms
#                             alpha = X/(4.0) + (n_hist-1.0)/2.0;
#                             beta  = Y/(no_of_histograms/n_hist) + (n_hist-1.0)/2.0;
#                             bin_index = ori/(360)*no_of_bins;
#                             if bin_index ==8:
#                                 bin_index= 7
#                             #  add contributions to respective bins in different histograms and in same histogram
#
#                             i0 = math.floor(alpha);
#                             j0 = math.floor(beta);
#
#                             for  x2  in range(max(0,int(i0)),min(int(i0)+1,int(n_hist)-1)+1) :
#                                 for y2 in range(max(0,int(j0)), min(int(j0)+1,int(n_hist)-1)+1):
#                                     # looping through all surrounding histograms.
#
#                                     k5 = 0
#                                     # Contribution to left bin.
#                                     k5 = (int(bin_index)+no_of_bins)%no_of_bins
#                                     desc[x2*n_hist*no_of_bins+y2*no_of_bins+k] += (1.0-(bin_index-math.floor(bin_index)))*(1.0-abs(float(x2)-alpha))*(1.0-abs(float(y2)-beta))*M
#
#                                     # Contribution to right bin.
#                                     k5 = (int(bin_index)+1+no_of_bins)%no_of_bins;
#                                     desc[x2*n_hist*no_of_bins+y2*no_of_bins+k] += (1.0-(math.floor(bin_index)+1-bin_index))*(1.0-abs(float(x2)-alpha))*(1.0-abs(float(y2)-beta))*M
#
#                                     #add the descriptor to the key point
#                 extPts3[i][j][k].append(desc)






##########################################################################################################################################################


    ######################################print the count of key points in each scale of each octave#################################
    # my_file = open("out1.txt", "w+")
    # print"writig started"
    # for i in range(0,octaves):
    #
    #     for j in range(0,2):
    #         print i,"  ",j,"  ",len(extPts2[i][j])
    #         temp1  = "octave"+str(i)+"  scale"+str(j)+"\n"
    #         my_file.write(temp1)
    #         for m in range(0,len(extPts2[i][j])):
    #             temp  = ''
    #             for n in range(0,12):
    #                 temp += str(extPts2[i][j][m][n])+"   "
    #                 temp += "\n"
    #             my_file.write(temp)
    #
    # my_file.close()
    # print "writing completed"
    ##################################################plot all key points###################################################
    c = 3

    plt.gray()
    #plt.title('key points after localization');
    plt.figure(2*number+3)
    plt.imshow(I)
    #plt.show()
    for i in range(0,octaves):
        for j in range(0,2):
            for p in range(0,len(extPts2[i][j])):
                #get the new x,y co-ordinates with respect to that octave scale
                #print "i",i,"j",j
                x1 = math.pow(2,i)*extPts2[i][j][p][1]
                y1 = math.pow(2,i)*extPts2[i][j][p][2]

                dx =  c*extPts2[i][j][p][3] * math.degrees(math.cos(extPts2[i][j][p][10]))
                dy =  c*extPts2[i][j][p][3] * math.degrees(math.sin(extPts2[i][j][p][10]))
                t1 = [x1]
                t2 = [y1]
                plt.plot(t2,t1,'r*')
                #plt the arrow
                #plt.arrow(x1,y1,dx,dy,color = 'yellow',head_width = 6,head_length = 7)
    ########################################################################################################################




image_name =['SIFT-input1.png','SIFT-input2.png']
for i in range(0,len(image_name)):
    sift(image_name[i],i)
plt.show()