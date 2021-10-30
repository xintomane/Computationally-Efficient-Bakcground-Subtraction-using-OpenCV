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
    KNN = cv.createBackgroundSubtractorKNN(100,400,False,)
    MOG2 = cv.createBackgroundSubtractorMOG2(300,400,False)
    MOG = cv.bgsegm.createBackgroundSubtractorMOG(300,4)
    

    
    

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
      

        opening_MOG = MOG.apply(opening)


        # show the current frame, foreground mask, subtracted result
        cv.imshow("Initial Frames", frame)
        cv.imshow("Foreground Masks MOG", fgMask_MOG)
        cv.imshow("Morphological Filter", opening_MOG)



        keyboard = cv.waitKey(30)
        if keyboard == ord('s'): 
            #cv.imwrite("frame.png",frame) 
            cv.imwrite("MOG.png",fgMask_MOG) 
            cv.imwrite("frame.png",frame)
            cv.imwrite("opening.png",opening_MOG)

        
        if keyboard == ord('q') or keyboard == 27:
            break


if __name__ == "__main__":
   
    # start BS-pipeline
    get_opencv_result(captured_video)
