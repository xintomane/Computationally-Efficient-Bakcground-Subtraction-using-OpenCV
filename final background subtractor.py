from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
import sys

captured_video = "BMC_dataset\Video_005.avi"

#kernel = np.ones((2,2),np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

def get_opencv_result(video_to_process):
    # create VideoCapture object for further video processing
    captured_video = cv.VideoCapture(video_to_process)
    # check video capture status
    if not captured_video.isOpened:
        print("Unable to open: " + video_to_process)
        exit(0)

    # instantiate background subtraction
    KNN = cv.createBackgroundSubtractorKNN(detectShadows=False)
    MOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    MOG = cv.bgsegm.createBackgroundSubtractorMOG()
    

    
    

    while True:
        # read video frames
        retval, frame = captured_video.read()

        # check whether the frames have been grabbed
        if not retval:
            break

        # resize video frames
        frame = cv.resize(frame, (640, 360))

        # pass the frame to the background subtractor
        fgMask_MOG = MOG.apply(frame)
        fgMask_MOG2 = MOG2.apply(frame)
        fgMask_KNN = KNN.apply(frame)
        

        opening = cv.morphologyEx(frame,cv.MORPH_OPEN,kernel,iterations = 2)
       # closing= cv.morphologyEx(frame,cv.MORPH_CLOSE,kernel,iterations = 2)
       # erode = cv.morphologyEx(frame, cv.MORPH_ERODE, kernel,iterations = 2)
        #dst = cv.GaussianBlur(frame,(3,3),cv.BORDER_DEFAULT)

        opening_MOG_before = MOG.apply(opening)
        frame_erode_before = cv.erode(frame, kernel, iterations=2)
        frame_dilate_before  = cv.dilate(frame_erode_before, kernel, iterations=2)
        separate_filters_before =MOG.apply(frame_dilate_before)
        #test4 = MOG.apply(dst)

        fgMask_erode = cv.erode(fgMask_MOG, kernel, iterations=2)
        fgMask_dilate = cv.dilate(fgMask_erode, kernel, iterations=2)

        fgMask_dilate[np.abs(fgMask_dilate) < 250] = 0

        #apply mrphological 'opening' filter
        opening_MOG = cv.morphologyEx(fgMask_MOG,cv.MORPH_OPEN,kernel)
        #opening_MOG2 = cv.morphologyEx(fgMask_MOG2,cv.MORPH_OPEN,kernel)
        #opening_KNN = cv.morphologyEx(fgMask_KNN,cv.MORPH_OPEN,kernel)

        #apply mrphological 'closing' filter
        closing_MOG = cv.morphologyEx(fgMask_MOG,cv.MORPH_CLOSE,kernel)
       
        median = cv.medianBlur(fgMask_MOG2,5)
        # show the current frame, foreground mask, subtracted result
        cv.imshow("Initial Frames", frame)
        cv.imshow("Foreground Masks MOG", fgMask_MOG)
        cv.imshow("Opening after BGS", opening_MOG)
        #cv.imshow("Foreground Masks MOG (Original)", fgMask_MOG)
        cv.imshow("Morphological filters after BGS",fgMask_dilate)
        cv.imshow("Opening before BGS", opening_MOG_before)
        cv.imshow("Morphological filters before BGS", separate_filters_before)
        #cv.imshow("Erode", test3)
        #cv.imshow("Gaussian Blur", test4)


        keyboard = cv.waitKey(30)
        if keyboard == ord('s'): 
            #cv.imwrite("frame.png",frame) 
            cv.imwrite("MOG.png",fgMask_MOG) 
            cv.imwrite("frame.png",frame)
            cv.imwrite("opening.png",opening_MOG)
            cv.imwrite("closing.png",closing_MOG)
            cv.imwrite("separate.png",fgMask_dilate)
        
        if keyboard == ord('q') or keyboard == 27:
            break


if __name__ == "__main__":
   
    # start BS-pipeline
    get_opencv_result(captured_video)