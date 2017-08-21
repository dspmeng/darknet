#!/bin/bash
set -e

./darknet detector train cfg/dss_barrier_all.data cfg/tiny-yolo-dss_barrier_all.cfg ../weights/darknet.conv.weights

# validate without bn
rm -f results/*

./darknet detector valid cfg/dss_barrier_all.data cfg/tiny-yolo-dss_barrier_all.cfg   \
    backup/tiny-yolo-dss_barrier_all_final.weights \
    -results results -testlist ../dss_barrier/test_all.txt

python scripts/run_ap.py -a ../dss_barrier/labels/ -i ../dss_barrier/test_all.txt -n ../dss_barrier/class_names.txt -d
