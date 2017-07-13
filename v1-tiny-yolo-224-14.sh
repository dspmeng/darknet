#!/bin/bash
set -e

TARGET="tiny-yolo-224-14"

# extract useful layers from darknet base network
./darknet partial cfg/yolov1/${TARGET}-base.cfg \
                  ../weights/darknet.conv.weights \
                  ../weights/darknet.conv-${TARGET}.weights 13

# fine tune
if [ ! -e backup-v1-${TARGET} ]; then
    mkdir backup-v1-${TARGET}
fi
FT_CFG=${TARGET}-transfer
./darknet yolo train cfg/yolov1/${FT_CFG}.cfg ../weights/darknet.conv-${TARGET}.weights \
    -backup backup-v1-${TARGET} 2>&1 | tee v1-${TARGET}.log

# merge bn layer with preceding conv layer
./darknet mergebn cfg/yolov1/${TARGET}.cfg \
    backup-v1-${TARGET}/${FT_CFG}_final.weights \
    backup-v1-${TARGET}/${TARGET}_final-nobn.weights

# validate without bn
rm -rf results-v1-${TARGET}
mkdir results-v1-${TARGET}

./darknet yolo valid cfg/yolov1/${TARGET}-nobn.cfg   \
    backup-v1-${TARGET}/${TARGET}_final-nobn.weights \
    -results results-v1-${TARGET}

python scripts/run_ap.py -r results-v1-${TARGET}
