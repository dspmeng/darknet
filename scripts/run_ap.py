from voc_eval import voc_eval, darknet_label_parser, voc_annot_parser
import os, sys
import argparse
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from itertools import cycle

marker = cycle((',', '+', 'o'))

ANNOTATION_PATH = '../pascal_voc/VOCdevkit/VOC2007/Annotations/{}.{}'
IMAGE_LIST_FILE = '../pascal_voc/2007_test.txt'
CLASS_NAMES = 'data/voc.names'
RESULT_PATH = 'results/comp4_det_test_{}.txt'

"""
voc test
$ rm results/*
$ rm -rf results-voc
$ ./darknet detector valid cfg/voc.data cfg/tiny-yolo-voc.cfg ../weights/tiny-yolo-voc.weights
$ cp -r results results-voc
$ python scripts/run_ap.py -r results-voc/

dss_barrier test
$ rm results/*
$ rm -rf results-dss_barrier
$ ./darknet detector valid cfg/dss_barrier.data cfg/tiny-yolo-dss_barrier.cfg ../dss_barrier/tiny-yolo-dss_barrier_final.weights
$ cp -r results results-dss_barrier
$ python scripts/run_ap.py -a ../dss_barrier/labels/ -i ../dss_barrier/test_all.txt -n ../dss_barrier/class_names.txt -r results-dss_barrier/ -d

tiny yolo voc on dss_barrier
$ rm results/*
$ rm -rf results-voc_on_dss_barrier
$ ./darknet detector valid cfg/voc_on_dss_barrier.data cfg/tiny-yolo-voc.cfg ../weights/tiny-yolo-voc.weights
$ cp -r results results-voc_on_dss_barrier
$ python scripts/run_ap.py -a ../dss_barrier/labels/ -i ../dss_barrier/test.txt -n ../dss_barrier/class_names.txt -r results-voc_on_dss_barrier/ -d -c car
"""

DRAW = ('car', 'bottle')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cls', help='calculate AP of specified class')
    parser.add_argument('-a', '--annotations', help='dir to annotations/labels')
    parser.add_argument('-i', '--images', help='list file of test image')
    parser.add_argument('-n', '--names', help='list file of class names')
    parser.add_argument('-r', '--results', help='dir to pascal voc results')
    parser.add_argument('-d', '--darknet', help='annotation in darknet format', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.annotations:
        ANNOTATION_PATH = ANNOTATION_PATH.replace('../pascal_voc/VOCdevkit/VOC2007/Annotations/', args.annotations)
    if args.images:
        IMAGE_LIST_FILE = args.images;
    if args.names:
        CLASS_NAMES = args.names 
    if args.results:
        RESULT_PATH = RESULT_PATH.replace('results', args.results)

    with open(CLASS_NAMES) as f:
        lines = f.readlines()
    classes = [l.strip() for l in lines]

    if args.darknet:
        rec_parser = darknet_label_parser(classes)
    else:
        rec_parser = voc_annot_parser()

    # remove annotation/label cache
    if os.path.isfile('./annots.pkl'):
        os.remove('./annots.pkl')

    if args.cls:
        recall, prec, ap = voc_eval(RESULT_PATH, ANNOTATION_PATH, IMAGE_LIST_FILE,
                                    args.cls, '.', parser=rec_parser)
        print 'AP of {}: {}'.format(args.cls, ap)
        plt.plot(recall, prec, label='{}: {:.2f}'.format(args.cls, ap))
    else:
        mean_ap = 0.0
        for cls in classes:
            recall, prec, ap = voc_eval(RESULT_PATH, ANNOTATION_PATH, IMAGE_LIST_FILE,
                                        cls, '.', parser=rec_parser)
            print 'AP of {}: {}'.format(cls, ap)
            if True:#cls in DRAW:
                plt.plot(recall, prec, label='{}: {:.2f}'.format(cls, ap),
                         linewidth=0.5, marker=marker.next())
            mean_ap += ap
        mean_ap /= len(classes)
        print 'Mean AP of VOC pascal {}'.format(mean_ap)
        plt.title('mAP: {:.2f}'.format(mean_ap))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig('ap.png')
    plt.show()
