#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os, sys, cv2, re
import argparse

CLASSES = (
    'other',
    'car',
    'mpv',
    'suv',
    'minibus',
    'mediumbus',
    'largebus',
    'minitruck',
    'largetruck')

cache_yolo = {}

def build_cache_yolo(log):
    regrex_img = re.compile(r'(\d+.jpg): Predicted')
    regrex_det = re.compile(r'(car|bus|truck),(\d+),(\d+),(\d+),(\d+)')
    with open(log) as f:
         for line in f:
             match = regrex_img.search(line)
             if match:
                 img = match.group(1)
                 cache_yolo[img] = []
             match = regrex_det.search(line)
             if match:
                 cache_yolo[img].append([
                     match.group(1),
                     int(match.group(2)),
                     int(match.group(3)),
                     int(match.group(4)),
                     int(match.group(5))])

def vis_detections(img, label, dets):
    with open(label) as l:
        gt = l.next() 

    im = cv2.imread(img)
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(dets)):
        bbox = dets[i][1:]
        cls = dets[i][0]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2], bbox[3], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(cls),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title('{}: {}'.format(img, gt), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('images', help='path to images')
    parser.add_argument('labels', help='path to labels')
    parser.add_argument('log', help='log of reference')
    parser.add_argument('--id', default='000000.jpg')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    build_cache_yolo(args.log)
    all_imgs = os.listdir(args.images)
    all_imgs.sort()
    start = all_imgs.index(args.id)
    for img in all_imgs[start:]:
        vis_detections(os.path.join(args.images, img),
                       os.path.join(args.labels, img.split('.')[0]+'.txt'),
                       cache_yolo[img])
