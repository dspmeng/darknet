#!/usr/bin/env python

"""
scripts/viz_validation.py --thresh 0.2 ~/place/barrier_ssd/test_list.txt ../dss_barrier/labels/ ~/deeplearning/ssd-caffe/jobs/VGGNet/barrier/SSD_300x300/results/comp4_det_test_largetruck.txt
"""

import os, sys, cv2
import argparse
from matplotlib import pyplot as plt


CLASS_NAMES = [
    'minibus',
    'minitruck',
    'car',
    'mediumbus',
    'mpv',
    'suv',
    'largetruck',
    'largebus',
    'other']


def viz_results(list_file, gt_dir, result_file, thresh):
    img_dict = {}
    images = [line.rstrip() for line in open(list_file).readlines()]
    for img in images:
        idx = os.path.basename(img).split('.')[0]
        img_dict[idx] = img

    labels = [line.rstrip() for line in open(result_file).readlines()]
    previous_idx = None
    objects = []
    for label in labels:
        idx, score, xmin, ymin, xmax, ymax = label.split()
        if idx != previous_idx:
            if objects:
                gts = [line.rstrip() for line in \
                       open(os.path.join(gt_dir, previous_idx+'.txt')).readlines()]
                show_results(img, gts, objects)
            previous_idx = idx
            img = img_dict[idx]
            del objects[:]
        score = float(score)
        if score < thresh:
            continue
        objects.append([int(float(xmin)), int(float(ymin)),
                        int(float(xmax)), int(float(ymax)), float(score)])


def show_results(image, gts, objects):
    im = cv2.imread(image)
    for gt in gts:
        t,cx,cy,w,h = gt.split()
        xmin = int((float(cx) - float(w)/2) * im.shape[1])
        ymin = int((float(cy) - float(h)/2) * im.shape[0])
        xmax = int(float(w) * im.shape[1]) + xmin
        ymax = int(float(h) * im.shape[0]) + ymin
        cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
        cv2.putText(im, CLASS_NAMES[int(t)],
                    ((xmax+xmin)/2, (ymax+ymin)/2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    for obj in objects:
        cv2.rectangle(im, (obj[0],obj[1]), (obj[2],obj[3]), (0,255,0), 3)
        cv2.putText(im, 'score: {}'.format(obj[4]),
                    (obj[0]+4, obj[3]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    im = im[:,:,::-1]
    plt.title(image)
    plt.imshow(im)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='visualize validate results with bbox overlay')
    parser.add_argument('list_file', help='list file of images')
    parser.add_argument('gt_dir', help='dir to ground truth labels')
    parser.add_argument('result_file', help='detection result file to check')
    parser.add_argument('--thresh', help='confidience score threshold',
                        type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    viz_results(args.list_file, args.gt_dir, args.result_file, args.thresh)
