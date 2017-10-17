#!/bin/bash

initial_weights=../weights/darknet.conv.weights
if [ ! -z $1 ]; then
    initial_weights=$1
fi

./darknet detector train cfg/lp.data cfg/tiny-yolo-lp.cfg ${initial_weights}
./darknet detector valid cfg/lp.data cfg/tiny-yolo-lp.cfg backup_lp/tiny-yolo-lp_final.weights
python scripts/run_ap.py -a /home/haobai/wintone/labels/ -i /home/haobai/wintone/test.txt -n /home/haobai/wintone/class_name.txt -d

