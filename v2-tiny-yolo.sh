#!/bin/bash

initial_weights=../weights/darknet.conv.weights
if [ ! -z $1 ]; then
    initial_weights=$1
fi

./darknet detector train cfg/voc.data cfg/tiny-yolo-voc.cfg ${initial_weights}
./darknet detector valid cfg/voc.data cfg/tiny-yolo-voc.cfg backup/tiny-yolo-voc_final.weights
python scripts/run_ap.py

