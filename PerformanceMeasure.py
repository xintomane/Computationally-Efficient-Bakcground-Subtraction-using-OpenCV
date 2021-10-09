from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
import tracemalloc
from memory_profiler import profile


input_video = "real\Video_005.avi"
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

@profile
def get_opencv_result(video_to_process):
    # create VideoCapture object for further video processing
    captured_video = cv.VideoCapture(video_to_process)
    # check video capture status
    if not captured_video.isOpened:
        print("Unable to open: " + video_to_process)
        exit(0)

    # instantiate background subtraction
    #GSOC = cv.bgsegm.createBackgroundSubtractorGSOC()
    #KNN = cv.createBackgroundSubtractorKNN()
    #MOG2 = cv.createBackgroundSubtractorMOG2()
    MOG = cv.bgsegm.createBackgroundSubtractorMOG()
    #LSBP = cv.bgsegm.createBackgroundSubtractorLSBP()
    #GMG = cv.bgsegm.createBackgroundSubtractorGMG(10, 0.8)
    #CNT = cv.bgsegm.createBackgroundSubtractorCNT()

    

    while True:
        # read video frames
        retval, frame = captured_video.read()

        # check whether the frames have been grabbed
        if not retval:
            break

        # resize video frames
        frame = cv.resize(frame, (640, 360))

        # pass the frame to the background subtractor
        frame = cv.morphologyEx(frame,cv.MORPH_OPEN,kernel,iterations = 2)
        fgMask = MOG.apply(frame)
        # obtain the background without foreground mask
        #background = MOG.getBackgroundImage()

        # show the current frame, foreground mask, subtracted result
        cv.imshow("Initial Frames", frame)
        cv.imshow("Foreground Masks", fgMask)
        #cv.imshow("Subtraction Result", background)

        keyboard = cv.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break


if __name__ == "__main__":
   
    # start BS-pipeline
    tracemalloc.start()
    get_opencv_result(input_video)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()