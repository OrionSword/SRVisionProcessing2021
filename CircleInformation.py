# Find all circles in a masked image and calculate their positions and
# distance to the camera.

# Written in Python 3.9.1 by Caleb Keller

import cv2
import math
import glob
import numpy as np

# Vars needed to calculate distance to camera (all measurements are in mm)
CAMERA_FOCAL_LENGTH = 4.6
CAMERA_SENSOR_HEIGHT = 4.55
OBJECT_REAL_HEIGHT = 177.8

# Redundant - needs improved
def pointDistance(X1, Y1, X2, Y2):
    return math.sqrt((X2-X1)**2 + (Y2-Y1)**2)

def circleOverlap(circleA, circleB, overlapThresh):
    # https://mathworld.wolfram.com/Circle-CircleIntersection.html

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

    if d + a <= b:
        # The smaller circle is completely inside the larger one. Just get its area.
        overlapArea =  math.pi * a2
    else:
        # Calculate the area of the shape formed by the intersection of the two circles.
        x = a2 * math.acos((d2 + a2 - b2) / (2 * d * a))
        z = b2 * math.acos((d2 + b2 - a2) / (2 * d * b))
        y = math.sqrt((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)) / 2
    
        overlapArea = x + z - y

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
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, 20, param1 = 140,
                               param2 = 40, minRadius = 10, maxRadius = 150)

    # HoughCircles returns none if there are no circles. To make
    # programming easier, return an empty array instead.
    if circles is None:
        return []
    else:
        circles = circles[0]

    # Compare all the circles in the array and ensure there are none
    # that overlap more than the threshold percentage. If two circles
    # overlap more than the threshold, remove the smaller one.

    cur = 0
    
    
    while cur != circles.shape[0]:
        for j in range(circles.shape[0] - 1, cur, -1):

            circleA = circles[cur]
            circleB = circles[j]

            if circleOverlap(circleA, circleB, OVERLAP_THRESHOLD):

                if (circleA[2] > circleB[2]):
                    circles = np.delete(circles, j, axis=0)
                else:
                    circles = np.delete(circles, cur, axis=0)
                    cur -= 1
                    break

        cur += 1
    

    return circles.tolist()

def distToCamera(circle, imHeight):
    '''
       "The ratio of the size of the object on the sensor and 
       the size of the object in real life is the same as the ratio 
       between the focal length and distance to the object" (Matt Grum).

       https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo
    '''
    return (CAMERA_FOCAL_LENGTH * OBJECT_REAL_HEIGHT * imHeight) / (circle[2] * 2 * CAMERA_SENSOR_HEIGHT)

# Directly copied from https://github.com/OrionSword/SRVisionProcessing2020/blob/master/StarGazer_Current.py
def calibrateCamera():
    #CAMERA CALIBRATION (from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((10*7,3), np.float32)
    square_size = 0.02 #in meters
    for i in range(0,7): #the y coordinate
        for j in range(0,10): #the x coordinate
            objp[i*10 + j][0] = j * square_size
            objp[i*10 + j][1] = i * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #get a list of all the calibration images from the folder
    calibration_images = glob.glob(r"Calibration_Images\*.jpg")

    for grid_img in calibration_images:
        img = cv2.resize(cv2.imread(grid_img), (960,454))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10,7),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
            print("Finished: ", grid_img)
        else:
            print("FAILED!!!! Chessboard corner finding failed on: ", grid_img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print(mtx[0][0], " ", mtx[1][1])

if __name__ == '__main__':
    frame = cv2.imread("186.png")

    upper_thresholds = np.array([38, 255, 255])
    lower_thresholds = np.array([19, 62, 158])

    im = cv2.resize(frame, (960,540))#downscale the frame as well to save processing power
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) #Convert BGR (RGB) to HSV
    mask_im = cv2.inRange(im_hsv, lower_thresholds, upper_thresholds)
    kernel = np.ones((5,5), np.uint8)
    mask_im = cv2.erode(mask_im, kernel, iterations=1)
    mask_im = cv2.dilate(mask_im, kernel, iterations=1)

    circles = getCircles(mask_im)

    print(circles)