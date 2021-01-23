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



#Load video
vid_source = cv2.VideoCapture(r"Sample Footage\20210121_193221.mp4")
total_frames = vid_source.get(cv2.CAP_PROP_FRAME_COUNT)
current_frame = 0
success = True

#Create window
cv2.namedWindow("Thresholds") #, cv2.WINDOW_AUTOSIZE
cv2.moveWindow("Thresholds", 0, 0)
cv2.resizeWindow("Thresholds", 500, 350)
cv2.createTrackbar("H High","Thresholds",38,255,lambda x:x) #38
cv2.createTrackbar("S High","Thresholds",255,255,lambda x:x) #255
cv2.createTrackbar("V High","Thresholds",255,255,lambda x:x) #255
cv2.createTrackbar("H Low","Thresholds",19,255,lambda x:x) #19
cv2.createTrackbar("S Low","Thresholds",62,255,lambda x:x) #62
cv2.createTrackbar("V Low","Thresholds",158,255,lambda x:x) #158

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

        #Read Image from video file
        #need to move the pointer back to beginning of the video if we reach the end (i.e. loop the video)
        if current_frame >= (total_frames-1):
            vid_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
        else:
            current_frame += 1

        success, frame = vid_source.read()
        if not success:
            print("FRAME CAPTURE FAILED")
            break
        
        im = cv2.resize(frame, (960,540))#downscale the frame as well to save processing power
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) #Convert BGR (RGB) to HSV

        start_time = time.time() #START TIMING
        
        mask_im = cv2.inRange(im_hsv, lower_thresholds, upper_thresholds)
        kernel = np.ones((5,5), np.uint8)
        mask_im = cv2.erode(mask_im, kernel, iterations=1)
        mask_im = cv2.dilate(mask_im, kernel, iterations=1)
        mask_im_bgr = cv2.cvtColor(mask_im, cv2.COLOR_GRAY2BGR)

        #Detect blobs
        """
        contour_data = cv2.findContours(mask_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL RETR_LIST

        contours = [  SRContour(contour, approx_poly_tolerance, 8) for contour in contour_data[0] if len(contour) >= min_points  ] #strips out the "contours" that have less than the specified number of points
        """
        
        end_time = time.time() #END TIMING

        #Draw lines
        im_with_keypoints = im.copy()
        """
        for contour in contours:
            im_with_keypoints = contour.draw(im_with_keypoints)
        """
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


