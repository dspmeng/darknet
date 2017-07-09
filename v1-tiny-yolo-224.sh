#!/bin/bash
set -e

TARGET="tiny-yolo-224"

if [ ! -e backup-v1-${TARGET} ]; then
    mkdir backup-v1-${TARGET}
fi

./darknet yolo train cfg/yolov1/${TARGET}.cfg ../weights/darknet.conv.weights \
    -backup backup-v1-${TARGET} 2>&1 | tee v1-${TARGET}.log

./darknet mergebn cfg/yolov1/${TARGET}.cfg \
    backup-v1-${TARGET}/${TARGET}_final.weights \
    backup-v1-${TARGET}/${TARGET}_final-nobn.weights

rm -rf results-v1-${TARGET}
mkdir results-v1-${TARGET}

./darknet yolo valid cfg/yolov1/${TARGET}-nobn.cfg   \
    backup-v1-${TARGET}/${TARGET}_final-nobn.weights \
    -results results-v1-${TARGET}

python scripts/run_ap.py -r results-v1-${TARGET}
