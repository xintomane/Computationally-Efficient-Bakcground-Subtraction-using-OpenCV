import numpy as np
import cv2 as cv
import argparse
import os
import glob

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


    path = './real\Video_001'
    path_dir = find_relevant_dirs(path)
    print(len(path_dir))
    for seq in path_dir:
        ground,frame = load_sequence(seq)

    
   

    bgs = cv.createBackgroundSubtractorMOG2()
    for i in range(len(ground)):
        frame[i] = np.uint8(cv.imread(frame[i], cv.IMREAD_COLOR))
        ground[i] = np.uint8(cv.imread(ground[i], cv.IMREAD_COLOR))
        cv.imshow('Frame', frame[i])
        cv.imshow('Ground-truth', ground[i])
        mask = bgs.apply(frame[i])
        bg = bgs.getBackgroundImage()
        cv.imshow('BG', bg)
        cv.imshow('Output mask', mask)
        k = cv.waitKey(0)
        if k == 27:
            break


if __name__ == '__main__':
    main()