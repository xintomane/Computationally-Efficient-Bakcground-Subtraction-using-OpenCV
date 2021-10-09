from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
import sys


captured_video = cv.VideoCapture("BMC_dataset\Video_007.avi")
#captured_video = cv.VideoCapture("highway.mp4") #for testing
kernel = np.ones((4,4),np.uint8)
#Adapted from https://github.com/AhadCove/Background-Subtractor-Comparisons/blob/master/main.py


GSOC = cv.bgsegm.createBackgroundSubtractorGSOC()
KNN = cv.createBackgroundSubtractorKNN()
MOG2 = cv.createBackgroundSubtractorMOG2()
MOG = cv.bgsegm.createBackgroundSubtractorMOG()
LSBP = cv.bgsegm.createBackgroundSubtractorLSBP()
GMG = cv.bgsegm.createBackgroundSubtractorGMG()
CNT = cv.bgsegm.createBackgroundSubtractorCNT()




while(1):
    retval, frame = captured_video.read()
    # check whether the frames have been grabbed
    if not retval:
        break
    

   
    frame = cv.resize(frame, (640, 360))

    GSOC_fgMask = GSOC.apply(frame)
    KNN_fgMask = KNN.apply(frame)
    MOG2_fgMask = MOG2.apply(frame)
    MOG_fgMask = MOG.apply(frame)
    LSBP_fgMask = LSBP.apply(frame)
    GMG_fgMask = GMG.apply(frame)
    CNT_fgMask = CNT.apply(frame)

    MOG_Count = np.count_nonzero(MOG_fgMask)
    kernel_2 = np.ones((MOG_Count,MOG_Count),np.uint8)


    opening_MOG = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_MOG2 = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_KNN = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_GSOC = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_LSBP = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_GMG = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    opening_CNT = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    #clsoing = cv.morphologyEx(MOG_fgMask,cv.MORPH_CLOSE,kernel)
    #erosion = cv.morphologyEx(MOG_fgMask,cv.MORPH_ERODE,kernel)



    

    cv.putText(GSOC_fgMask, 'GSOC', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)
    cv.putText(KNN_fgMask, 'KNN', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)
    cv.putText(MOG2_fgMask, 'MOG2', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)                      
    cv.putText(MOG_fgMask, 'MOG', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)
    cv.putText(LSBP_fgMask, 'LSBP', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)
    cv.putText(GMG_fgMask, 'GMG', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)
    cv.putText(CNT_fgMask, 'CNT', (100, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2 ,(255,255,255),2, cv.LINE_AA)

    cv.imshow('Frame', frame)
    cv.imshow('GSOC', GSOC_fgMask)
    cv.imshow('KNN', KNN_fgMask)
    cv.imshow('MOG', MOG_fgMask)
    cv.imshow('MOG2', MOG2_fgMask)
    cv.imshow('LSBP', LSBP_fgMask)
    cv.imshow('GMG', GMG_fgMask)
    cv.imshow('CNT', CNT_fgMask)
    #cv.imshow("Opening",opening)
    #cv.imshow("Closing",clsoing)
    #cv.imshow('erosion',erosion)
   # print(MOG_Count)



    keyboard = cv.waitKey(30)
    if keyboard == ord('s'):
        cv.imwrite("final_results\MOG_v2.png",MOG_fgMask) 
        cv.imwrite("final_results\MOG2_v2.png",MOG2_fgMask) 
        cv.imwrite("final_results\GSOC_v2.png",GSOC_fgMask) 
        cv.imwrite("final_results\KNN_v2.png",KNN_fgMask) 
        cv.imwrite("final_results\CNT_v2.png",CNT_fgMask) 
        cv.imwrite("final_results\LSBP_v2.png",LSBP_fgMask) 
        cv.imwrite("final_results\GMG_v2.png",GMG_fgMask) 
        cv.imwrite("final_results\FRAME_v2.png",frame) 
        
    if keyboard == ord('q') or keyboard == 27:
        break

captured_video.release()
cv.destroyAllWindows()





               
 
