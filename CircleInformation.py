# Find all circles in a masked image and calculate their positions and
# distance to the camera.

# Written in Python 3.9.1 by Caleb Keller

import cv2
import math

# Vars needed to calculate distance to camera
CAMERA_FOCAL_LENGTH = 0
CAMERA_SENSOR_HEIGHT = 0
OBJECT_REAL_HEIGHT = 0

# Redundant - needs improved
def pointDistance(X1, Y1, X2, Y2):
    return math.sqrt((X2-X1)**2 + (Y2-Y1)**2)

def circleOverlap(circleA, circleB, overlapThresh):
    # Get radii and distance between centers
    a = min(circleA[2], circleB[2])
    b = max(circleA[2], circleB[2])
    d = pointDistance(circleA[0], circleA[1], circleB[0], circleB[1])
    
    # If the distance between the centers of the circles is larger than
    # the sum of their radii, they are not intersecting.
    if d >= a + b:
        return False

    # Save squares of radii and distance to avoid calculating repeatedly.
    a2 = math.pow(a, 2)
    b2 = math.pow(b, 2)
    d2 = math.pow(d, 2)

    if (d <= b - a):
        # The smaller circle is completely inside the larger one. Just get its area.
        overlapArea =  math.pi * a2
    else:
        # Calculate the area of the shape formed by the intersection of the two circles.
        x = (a2 - b2 + d2) / (2 * d)
        z = x * x
        y = math.sqrt(a2 - z)
    
        overlapArea = a2 * math.asin(y / a) + b2 * math.asin(y / b) - y * (x + math.sqrt(z + b2 - a2))

    # The percentage of overlap is the ratio of the area of the
    # overlapping area to the area of the smaller circle.
    overlapPercent = overlapArea / (math.pi * a2)

    return overlapPercent > overlapThresh


def getCircles(inp):
    # Tuneable constant
    OVERLAP_THRESHOLD = 0.5

    # Blur the image
    image = cv2.GaussianBlur(inp, (7, 7), 1.5)

    # Finds all circles in image
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,
                               param2 = 25, minRadius = 0, maxRadius = 0)

    # HoughCircles returns none if there are no circles. To make
    # programming easier, return an empty array instead.
    if circles is None:
        return [[]]

    # Compare all the circles in the array and ensure there are none
    # that overlap more than the threshold percentage. If two circles
    # overlap more than the threshold, remove the smaller one.
    for i in range(0, circles[0].size, -1):
        for j in range(0, i - 1, -1):
            circleA = circles[0][i]
            circleB = circles[0][j]

            if circleOverlap(circleA, circleB, OVERLAP_THRESHOLD):
                print("Overlap detected")
                if (circleA[2] > circleB[2]):
                    del circles[j]
                else:
                    del circles[i]
    return circles

def distToCamera(circle, imHeight):
    '''
       "The ratio of the size of the object on the sensor and 
       the size of the object in real life is the same as the ratio 
       between the focal length and distance to the object" (Matt Grum).

       https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
    '''
    return (CAMERA_FOCAL_LENGTH * OBJECT_REAL_HEIGHT * imHeight) / (circle[2] * 2 * CAMERA_SENSOR_HEIGHT)



#print(circleOverlap([0, 0, 10], [10, 0, 2], 0.3))