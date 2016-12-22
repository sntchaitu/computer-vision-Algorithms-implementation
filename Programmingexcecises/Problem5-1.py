from __future__ import division
import math
import matplotlib.pyplot as plt


def ant_simulation(time_interval, speed):
    """
        below function takes time_interval(between 2 simulation steps) and no_of_steps for total simulation and plots the
         position of each ant for each time step until the distance between between two ants is < distance (speed*time_interval)
        Each ant slope differ with respect to its right ant
        distance between each ant is 1 mile we assume  1 unit of distance on graph as 1 mile
        Hence initially  ant A1 is at (0,0) A2 at(0,1) and A3 is at (1/2,srqt(3)/2) position
        three ants collide at the centroid of the triangle and all the ants will move on smooth curved paths.

    """
    """
    :param time_interval:
    :param no_of_steps:
    :return plots the position of each ant on matplotlib
    """

    # stores the initial positions of each ant
    x1 = 0
    y1 = 0
    x2 = 1
    y2 = 0
    x3 = 1 / 2
    y3 = math.sqrt(3) / 2
    dis = speed * time_interval
    # x axis length in the graph is from -1 to 2 and Y axis is from -1 to 2
    plt.axis([-1, 2, -1, 2])
    plt.plot([x1, x2, x3], [y1, y2, y3], 'ro')

    #simulation will run until the difference of distance between 2 ants and than the  step distance is > 0.061
    #when the distance between 2 points becomes very less the slope will increase in opposite direction and they started to move away.hence min distance
    #of 0.061 is is added to dis by means of  hit and trail and found that till tha distance they shrink forming equilateral triangle.
    while math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))>= dis+0.061:
        # calculates the slope of first ant with respect to its right ant
        m1 = (math.atan((y2 - y1) / (x2 - x1)))
        # ant 2 will differ from ant 1 in slope tems by 2*pi /3 and ant3 will differ from ant 2 in terms of slope by 2*pi/3.i.e,
        # with respect to ant 1 ant2 slope will differ by 4 * pi/3
        m2 = m1+(2*math.pi)/3
        m3 = m1+(4*math.pi)/3

        # calculates the new position of each ant from its old position

        x1new = x1 + dis * math.cos(m1)
        y1new = y1 + dis * math.sin(m1)
        x2new = x2 + dis * math.cos(m2)
        y2new = y2 + dis * math.sin(m2)
        x3new = x3 + dis * math.cos(m3)
        y3new = y3 + dis * math.sin(m3)

        # plots the new position on the graph
        plt.plot([x1new, x2new, x3new], [y1new, y2new, y3new], 'ro')

        # assigns  the new position to old postion as the new position will become old position for the next time step
        x1 = x1new
        y1 = y1new
        x2 = x2new
        y2 = y2new
        x3 = x3new
        y3 = y3new


    # shows the plot

    plt.show()


    # By  means of trial and error for the time slot value of 0.01 three ants move towards
    #  centroid in shrinking manner along smooth curve.
ant_simulation(0.01, 1)
