import numpy as np
import cv2 as cv
import argparse
import os
import glob

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
GSOC = cv.bgsegm.createBackgroundSubtractorGSOC()
KNN = cv.createBackgroundSubtractorKNN(100,400,True)
MOG2 = cv.createBackgroundSubtractorMOG2(300,400,False)
MOG = cv.bgsegm.createBackgroundSubtractorMOG(300)
LSBP = cv.bgsegm.createBackgroundSubtractorLSBP(nSamples=20,LSBPRadius=16,Tlower=2.0,Tupper=32.0,Tinc= 1.0, Tdec= 0.05, Rscale= 10.0, Rincdec=0.005, LSBPthreshold=8)
GMG = cv.bgsegm.createBackgroundSubtractorGMG(10,.8)
CNT = cv.bgsegm.createBackgroundSubtractorCNT()
{'nSamples': 20, 'LSBPRadius': 16, 'Tlower': 2.0, 'Tupper': 32.0, 'Tinc': 1.0, 'Tdec': 0.05, 'Rscale': 10.0, 'Rincdec': 0.005, 'LSBPthreshold': 8}
def main():

    def contains_relevant_files(root):
        return os.path.isdir(os.path.join(root, 'groundtruth')) and os.path.isdir(
        os.path.join(root, 'input'))


    def find_relevant_dirs(root):
        relevant_dirs = []
        for d in sorted(os.listdir(root)):
            d = os.path.join(root, d)
            if os.path.isdir(d):
                if contains_relevant_files(d):
                    relevant_dirs += [d]
                else:
                    relevant_dirs += find_relevant_dirs(d)
        return relevant_dirs

    def load_sequence(root):
        gt_dir, frames_dir = os.path.join(root, 'groundtruth'), os.path.join(root, 'input')
        gt = sorted(glob.glob(os.path.join(gt_dir, '*.bmp')))
        f = sorted(glob.glob(os.path.join(frames_dir, '*.bmp')))
        assert(len(gt) == len(f))
        return gt, f


    path = './real\Video_005'
    path_dir = find_relevant_dirs(path)
    print(len(path_dir))
    for seq in path_dir:
        ground,frame = load_sequence(seq)

    
   

    bgs = cv.createBackgroundSubtractorMOG2()
    for i in range(len(ground)):
        
        frame[i] = np.uint8(cv.imread(frame[i], cv.IMREAD_COLOR))
        frame[i]= cv.morphologyEx(frame[i],cv.MORPH_OPEN,kernel,iterations = 2)
        ground[i] = np.uint8(cv.imread(ground[i], cv.IMREAD_COLOR))
        
        cv.imshow('Frame', frame[i])
        cv.imshow('Ground-truth', ground[i])
       # mask = bgs.apply(frame[i])
        GSOC_fgMask = GSOC.apply(frame[i])
        KNN_fgMask = KNN.apply(frame[i])
        MOG2_fgMask = MOG2.apply(frame[i])
        MOG_fgMask = MOG.apply(frame[i])
        LSBP_fgMask = LSBP.apply(frame[i])
        GMG_fgMask = GMG.apply(frame[i])
        CNT_fgMask = CNT.apply(frame[i])
        #bg = bgs.getBackgroundImage()
        #cv.imshow('BG', bg)
        # cv.imshow('Frame', frame)
        cv.imshow('GSOC', GSOC_fgMask)
        cv.imshow('KNN', KNN_fgMask)
        cv.imshow('MOG', MOG_fgMask)
        cv.imshow('MOG2', MOG2_fgMask)
        cv.imshow('LSBP', LSBP_fgMask)
        cv.imshow('GMG', GMG_fgMask)
        cv.imshow('CNT', CNT_fgMask)
        k = cv.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()
