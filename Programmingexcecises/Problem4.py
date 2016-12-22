from __future__ import division
import random
import math


def compute_probability(value):

    """below function generate 1 million random points and check if each point
        falls in region 1 or 3  counts the number of times it happens

        assumptions: A  square region  with unit area in each quadrant is considered with co-ordinate axes as origin
        boundaries are  (1,1) ,  (-1,1) ,  (1,-1) and  (-1,-1)

        points  A = (0,1)  C = (0,1) and B = (1,1)
        random point  P= xi,yi  and origin O  = (0,0)
        Line L1 is  X+Y = 1

         Below are the conditions for a random point to be in any  of the regions

                 for region 1   -1<=Xi<0 and -1<Yi<1.Points that fall on line X = 0 are not  to be part of region 1

                 for region 3    generate a line L2 parallel to line L1  and passing through point P and if  perpendicular distance(dis2) of Origin  to line L2 is less than
                                 perpendicular distance of Origin  to line L1(dis1) then it falls in region 3.
                                 Points that fall on line L1 or  co-ordinates ( X = 0 and y>0)  or co-ordinates ( Y = 0 and x>0) are considered in this region
                                 Also points that fall on line X+Y =1 are considered in this region

                 for region 2    generate a line L2 parallel to line L1  and passing through point P and if  perpendicular distance(dis2) of Origin  to line L2 is greater than
                                 perpendicular distance of Origin  to line L1(dis1) then it falls in region 2.
                                 Points  with co-ordinates  (0<x<=1 and y==1)  or (x==1 and 0<y<=1) are considered in this region


                 for region 4   0<=X<=1 and -1<=Y<0.Points that fall on  co-ordinates  (x = 0  and y<0) are considered in this region.
                 Point fall on co-ordinates (y=0 and x>0) are part of this region

    """

    """
        points inferred: As the no of trials
    """

    # print (0.02/0.78)
    count = 0
    xlow = -1
    xhigh = 1
    ylow  = -1
    yhigh = 1

    num = 0 # counts the no of times (xi,yi) falls  into odd number region (region 1 or region 3)

    while count<value:
        x = random.uniform(xlow,xhigh)
        y = random.uniform(ylow,yhigh )

        # print('\n')
        #print (x,y)

        """generate a line L2 which passes through P(x,y) and parallel to line L1 Y = -X + 1

        """

        # slope of line L1
        m1 = -1

        # equation of line with Point P(x,y) and slope m1
        b = y - m1*x

        # compute distance from origin to Line L2 from origin (0,0) m1*x -y +b = 0
        dis2 = abs(m1*0 - 1*0 + b)/math.sqrt((m1 * m1) + (-1 * -1))

        # compute distance from Origin to Line L1 x+y -1 = 0
        dis1 = abs(1)/math.sqrt((1*1) + (1*1))

        #  conditional check for region 1
        if (x >= -1 and x<0) and (y > -1 and y< 1):
            num += 1
            count += 1
        elif ( x>0 and x < 1) and ( y > -1 and y < 0):  # conditional check for region 4
            count += 1
        # conditional check for region 3
        elif (x >=0  and x <= 1)and (y >= 0 and y <= 1) and (x + y == 1) and dis2 < dis1:
            num += 1
            count += 1
        # conditional check for region 2
        elif (x > 0  and x <= 1)and (y > 0 and y <= 1) and (x + y != 1) and dis2 > dis1:
            count += 1
    # prints the probability of (Xi,Yi) to be in odd number region
    print "probability to be in odd number region is:", num/value


compute_probability(10000000)










