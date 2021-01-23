#Installed CV2 library from instructions at:
#https://pypi.org/project/opencv-python/
#Used "pip3 install opencv-python" in CMD
#created with Python 3.6 on 04-22-2020 by Orion DeYoe

#Standard imports
import cv2 #OpenCV for Python library
import numpy as np
import math
import time
import copy
import glob

KEY_ESCAPE = 27

def PointDistance(X1, Y1, X2, Y2):
    return math.sqrt((X2-X1)**2 + (Y2-Y1)**2)

def PointDirection(X1, Y1, X2, Y2): #in radians
    DX = X2 - X1
    DY = Y2 - Y1
    
    if ((DX >= 0) and (DY == 0)):
        return 0

    elif ((DX > 0) and (DY > 0)):
        return math.atan(DY / DX)

    elif ((DX == 0) and (DY > 0)):
        return math.pi * 0.5

    elif ((DX < 0) and (DY > 0)):
        return math.atan(DY / DX) + math.pi

    elif ((DX < 0) and (DY == 0)):
        return math.pi

    elif ((DX < 0) and (DY < 0)):
        return math.atan(DY / DX) + math.pi

    elif ((DX == 0) and (DY < 0)):
        return math.pi * 1.5

    elif ((DX > 0) and (DY < 0)):
        return math.atan(DY / DX) + 2 * math.pi

def Radians(ang):
    return ang * math.pi / 180

def Degrees(ang):
    return ang * 180 / math.pi

def DrawClosedContour(image, contour, color=(0,0,255), thiccness=1):
        last_point = None
        for point in contour:
                if last_point is not None:
                        image = cv2.line(image,(last_point[0],last_point[1]),(point[0],point[1]),color,thiccness)
                last_point = point
        image = cv2.line(image,(last_point[0],last_point[1]),(contour[0][0],contour[0][1]),color,thiccness)
        return image

class SRContour:
    def __init__(self, point_data, simplify_tolerance, target_simplify_points, tol_sweep_start = 1, tol_sweep_end = 10, tol_sweep_step = 1):
        self.point_data = point_data
        self.points = [ [point[0][0],point[0][1]] for point in point_data ]
        self.left,self.top,self.width,self.height = cv2.boundingRect(self.point_data)
        self.right = self.left + self.width
        self.bottom = self.top + self.height
        
        self.simple_tolerance = None
        self.simple_contour = None

        for i in range(tol_sweep_start, tol_sweep_end+1, tol_sweep_step):
            simple_data = cv2.approxPolyDP(self.point_data, i, True)
            if len(simple_data) == target_simplify_points:
                self.simple_contour = [[point[0][0],point[0][1]] for point in simple_data]
                self.simple_tolerance = i
                break

        self.left_tape = [] #[  [x1,y1],  [x2,y2]  ]
        self.right_tape = []
        self.bottom_tape = []
        self.pairs = []
        self.centerpoints = []
        self.upper_left_point = []
        self.lower_left_point = []
        self.upper_right_point = []
        self.lower_right_point = []

        if self.simple_contour is not None:
            pts = copy.deepcopy(self.simple_contour)
            while len(pts) > 1: #loop until there isn't a pair of points left in the list (should empty the list for even numbers of points)
                pt1 = pts.pop(0) #pop out a point to pair another point with
                record = math.inf #this is necessary for making sure the first point we find is saved as the current record
                rec_ind = None #variable to store the record-holding index in
                for i in range(len(pts)): #loop through the rest of the points in the list (after popping one out earlier)
                    dist = PointDistance(pt1[0],pt1[1],pts[i][0],pts[i][1])
                    if dist < record: #save this point as the record holder if it is closer to the first point
                        record = dist
                        rec_ind = i
                pt2 = pts.pop(rec_ind) #pop the record holder out of the point list
                self.pairs.append([pt1,pt2]) #add the newly married couple (point pair) to the point list
            for pair in self.pairs:
                self.centerpoints.append([  int((pair[0][0]+pair[1][0])/2),  int((pair[0][1]+pair[1][1])/2)  ])
            #sort centerpoints by X coordinate
            x_sorted = sorted(self.centerpoints, key = lambda x: x[0])
            left_centers = sorted(x_sorted[0:int(len(x_sorted)/2)], key = lambda x: x[1])
            right_centers = sorted(x_sorted[int(len(x_sorted)/2):], key = lambda x: x[1])
            self.upper_left_point = left_centers[0]
            self.lower_left_point = left_centers[-1]
            self.upper_right_point = right_centers[0]
            self.lower_right_point = right_centers[-1]
            
            

    def draw(self, image):
        out_img = DrawClosedContour(image, self.points, (0,0,255), 2)
        #for pair in self.pairs:
        #    out_img = cv2.line(out_img,(pair[0][0],pair[0][1]),(pair[1][0],pair[1][1]),(0,242,255),2)
        if self.simple_contour is not None:
            out_img = cv2.line(out_img,(self.upper_left_point[0],self.upper_left_point[1]),(self.lower_left_point[0],self.lower_left_point[1]),(0,242,255),2)
            out_img = cv2.line(out_img,(self.lower_left_point[0],self.lower_left_point[1]),(self.lower_right_point[0],self.lower_right_point[1]),(0,242,255),2)
            out_img = cv2.line(out_img,(self.lower_right_point[0],self.lower_right_point[1]),(self.upper_right_point[0],self.upper_right_point[1]),(0,242,255),2)
        
        if self.simple_contour is not None:
            out_img = DrawClosedContour(out_img, self.simple_contour, (255,0,0), 1)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (self.left,self.bottom+20)
            fontScale              = .5
            fontColor              = (255,255,255)
            lineType               = 1

            out_img = cv2.putText(out_img,"Tol: "+str(self.simple_tolerance), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
        return out_img


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
calibration_images = glob.glob(r"Phone Images 06102020\Calibration Grid\*.jpg")

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




# Read image
im = cv2.resize(cv2.imread(r"Phone Images 06102020\Vision Target\20200610_152132.jpg", cv2.IMREAD_COLOR), (960,454))

#Convert BGR (RGB) to HSV
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

#Create window
cv2.namedWindow("Thresholds") #, cv2.WINDOW_AUTOSIZE
cv2.moveWindow("Thresholds", 0, 0)
cv2.resizeWindow("Thresholds", 500, 350)
cv2.createTrackbar("H High","Thresholds",255,255,lambda x:x) #102
cv2.createTrackbar("S High","Thresholds",38,255,lambda x:x) #255
cv2.createTrackbar("V High","Thresholds",255,255,lambda x:x) #255
cv2.createTrackbar("H Low","Thresholds",0,255,lambda x:x) #74
cv2.createTrackbar("S Low","Thresholds",0,255,lambda x:x) #150
cv2.createTrackbar("V Low","Thresholds",228,255,lambda x:x) #129

cv2.namedWindow("Images")
cv2.moveWindow("Images", 700, 0)

#is it loop time brother
min_points = 3
approx_poly_tolerance = 5
cont = True
while cont:
        #start_time = time.time() #START TIMING
        #Threshold image
        h_upper_threshold = cv2.getTrackbarPos("H High", "Thresholds")
        s_upper_threshold = cv2.getTrackbarPos("S High", "Thresholds")
        v_upper_threshold = cv2.getTrackbarPos("V High", "Thresholds")
        h_lower_threshold = cv2.getTrackbarPos("H Low", "Thresholds")
        s_lower_threshold = cv2.getTrackbarPos("S Low", "Thresholds")
        v_lower_threshold = cv2.getTrackbarPos("V Low", "Thresholds")
        
        upper_thresholds = np.array([h_upper_threshold,s_upper_threshold,v_upper_threshold])
        lower_thresholds = np.array([h_lower_threshold,s_lower_threshold,v_lower_threshold])

        start_time = time.time() #START TIMING
        mask_im = cv2.inRange(im_hsv, lower_thresholds, upper_thresholds)
        kernel = np.ones((5,5), np.uint8)
        mask_im = cv2.erode(mask_im, kernel, iterations=1)
        mask_im = cv2.dilate(mask_im, kernel, iterations=1)
        mask_im_bgr = cv2.cvtColor(mask_im, cv2.COLOR_GRAY2BGR)

        #Detect blobs
        contour_data = cv2.findContours(mask_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL RETR_LIST

        contours = [  SRContour(contour, approx_poly_tolerance, 8) for contour in contour_data[0] if len(contour) >= min_points  ] #strips out the "contours" that have less than the specified number of points
        
        #Solve PNP
        world_points = np.array([(0,-.469,2.494),(0,-0.235,2.088),(0,0.235,2.088),(0,0.469,2.494)], np.float32)
        image_points = np.array([np.array([contours[0].upper_left_point], np.float32),
                                 np.array([contours[0].lower_left_point], np.float32),
                                 np.array([contours[0].lower_right_point], np.float32),
                                 np.array([contours[0].upper_right_point], np.float32)]
                                , np.float32)
        _, rotation_vector, translation_vector = cv2.solvePnP(world_points, image_points, mtx, dist)
        end_time = time.time() #END TIMING

        #Draw lines
        im_with_keypoints = im.copy()
        for contour in contours:
            im_with_keypoints = contour.draw(im_with_keypoints)
        #end_time = time.time() #END TIMING

        #Draw frame time
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,30)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 1

        im_with_keypoints = cv2.putText(im_with_keypoints,"Time: "+str(end_time - start_time), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        #Display the images
        combined_im = cv2.vconcat([im_with_keypoints, mask_im_bgr])
        
        cv2.imshow("Images", combined_im)
        
        if cv2.waitKey(20) == KEY_ESCAPE: #cv2.waitKey(20) & 0xFF == ord('q')
            break


#CLEAN UP
cv2.destroyAllWindows()


