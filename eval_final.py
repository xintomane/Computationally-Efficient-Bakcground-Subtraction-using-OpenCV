import argparse
import cv2 as cv
import glob
import numpy as np
import os
import time
import csv


kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))



# https://github.com/opencv/opencv_contrib/blob/master/modules/bgsegm/samples/evaluation.py
#ALGORITHMS_TO_EVALUATE = [
#    (cv.createBackgroundSubtractorMOG2(300,400,False), "MOG2"),
#    (cv.createBackgroundSubtractorKNN(100,400,False), "KNN"),
#    (cv.bgsegm.createBackgroundSubtractorMOG(300), 'MOG'),
#    (cv.bgsegm.createBackgroundSubtractorGMG(10,.8), 'GMG'),
#    (cv.bgsegm.createBackgroundSubtractorGSOC(), "GSOC"),
#    (cv.bgsegm.createBackgroundSubtractorLSBP(nSamples=10,LSBPRadius=16,Tlower=2.0,Tupper=32.0,Tinc= 1.0, Tdec= 0.05, Rscale= 10.0, Rincdec=0.005, LSBPthreshold=8), "LSBP"),
#    (cv.bgsegm.createBackgroundSubtractorCNT(), 'CNT'),
    
#]

ALGORITHMS_TO_EVALUATE = [
    (cv.createBackgroundSubtractorMOG2(history=100,detectShadows=False), "MOG2"),
    (cv.createBackgroundSubtractorKNN(history=300,detectShadows=False), "KNN"),
    (cv.bgsegm.createBackgroundSubtractorMOG(), 'MOG'),
    #(cv.bgsegm.createBackgroundSubtractorGMG(150,.8), 'GMG'),
    #(cv.bgsegm.createBackgroundSubtractorGSOC(), "GSOC"),
    #(cv.bgsegm.createBackgroundSubtractorLSBP(nSamples=20,LSBPRadius=16,Tlower=2.0,Tupper=32.0,Tinc= 1.0, Tdec= 0.05, Rscale= 10.0, Rincdec=0.005, LSBPthreshold=8), "LSBP"),
    #(cv.bgsegm.createBackgroundSubtractorCNT(minPixelStability = 25), 'CNT'),
    
]

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
    gt = sorted(glob.glob(os.path.join(gt_dir, '*.png'))) #png for cd.net dataset bmp for BMC dataset
    f = sorted(glob.glob(os.path.join(frames_dir, '*.jpg'))) #jpg for cd.net dataset bmp for BMC dataset
    assert(len(gt) == len(f))
    return gt, f

def evaluate_algorithm(gt, frames, algo):
    #opening = cv.morphologyEx(MOG_fgMask,cv.MORPH_OPEN,kernel)
    #MOG = cv.bgsegm.createBackgroundSubtractorMOG()
    mask = []
    t_start = time.time()

    for i in range(len(gt)):
        frame = np.uint8(cv.imread(frames[i], cv.IMREAD_COLOR))
        frame = cv.morphologyEx(frame,cv.MORPH_OPEN,kernel,iterations = 2)
        mask.append(algo.apply(frame))

    average_duration = (time.time() - t_start) / len(gt)
    average_precision, average_recall, average_f1, average_accuracy = [], [], [], []

    for i in range(len(gt)):
        gt_mask = np.uint8(cv.imread(gt[i], cv.IMREAD_GRAYSCALE))
        roi = ((gt_mask == 255) | (gt_mask == 0))
        if roi.sum() > 0:
            gt_answer, answer = gt_mask[roi], mask[i][roi]

            tp = ((answer == 255) & (gt_answer == 255)).sum()
            tn = ((answer == 0) & (gt_answer == 0)).sum()
            fp = ((answer == 255) & (gt_answer == 0)).sum()
            fn = ((answer == 0) & (gt_answer == 255)).sum()
            #print(tp,fp)

            if tp + fp > 0:
                average_precision.append(float(tp) / (tp + fp))
            if tp+fp == 0:
                average_precision.append(0)    
            if tp + fn > 0:
                average_recall.append(float(tp) / (tp + fn))
            if tp + fn + fp > 0:
                average_f1.append(2.0 * tp / (2.0 * tp + fn + fp))
                average_accuracy.append(float(tp + tn) / (tp + tn + fp + fn))

    return (average_duration, 
            np.mean(average_precision),
            np.mean(average_recall),
            np.mean(average_f1),
            np.mean(average_accuracy))

def evaluate_on_sequence(seq, summary):
    gt, frames = load_sequence(seq)
    category, video_name = os.path.basename(os.path.dirname(seq)), os.path.basename(seq)
    print('=== %s:%s ===' % (category, video_name))

    for algo, algo_name in ALGORITHMS_TO_EVALUATE:
        #print('Algorithm name: %s' % algo_name)
        sec_per_step, precision, recall, f1, accuracy = evaluate_algorithm(gt, frames, algo)
        #print('Average accuracy: %.3f' % accuracy)
        #print('Average precision: %.3f' % precision)
        #print('Average recall: %.3f' % recall)
        #print('Average F1: %.3f' % f1)
        #print('Average sec. per step: %.4f' % sec_per_step)
        #print('')

        if category not in summary:
            summary[category] = {}
        if algo_name not in summary[category]:
            summary[category][algo_name] = []
        summary[category][algo_name].append((precision, recall, f1, accuracy,sec_per_step))

def main():
    parser = argparse.ArgumentParser(description='Evaluate all background subtractors using Change Detection 2014 dataset')
    #parser.add_argument('--dataset_path', help='Path to the directory with dataset. It may contain multiple inner directories. It will be scanned recursively.', required=True,)
    parser.add_argument('--algorithm', help='Test particular algorithm instead of all.')
    
    args = parser.parse_args()
    dataset_path='./baseline'
    
    dataset_dirs = find_relevant_dirs(dataset_path)
    print(len(dataset_dirs))
    assert len(dataset_dirs) > 0, ("Passed directory must contain at least one sequence from the Change Detection dataset. There is no relevant directories in %s. Check that this directory is correct." % (dataset_path))
    if args.algorithm is not None:
        global ALGORITHMS_TO_EVALUATE
        ALGORITHMS_TO_EVALUATE = filter(lambda a: a[1].lower() == args.algorithm.lower(), ALGORITHMS_TO_EVALUATE)
    summary = {}
    
    

    for seq in dataset_dirs:
        evaluate_on_sequence(seq, summary)
    

    for category in summary:
        for algo_name in summary[category]:
            summary[category][algo_name] = np.mean(summary[category][algo_name], axis=0)
    
    #for category in summary:
    #    print('=== SUMMARY for %s (Precision, Recall, F1, Accuracy,Duration) ===' % category)
        
    #    for algo_name in summary[category]:
    #        print('%05s: %.3f %.3f %.3f %.3f %.4f' % ((algo_name,) + tuple(summary[category][algo_name]))) 
    with open('eval_results.txt', 'w') as f:

        for category in summary:
            f.write('=== SUMMARY for %s (Precision, Recall, F1, Accuracy,Duration) ===' % category)
            f.write('\n')
            for algo_name in summary[category]:
                f.write('%05s: %.3f %.3f %.3f %.3f %.4f' % ((algo_name,) + tuple(summary[category][algo_name]))) 
                f.write('\n')
        
    



if __name__ == '__main__':
    main()


