#!/bin/bash

PWD=`pwd`
folder="/home/jiameng/hackspace/dss-data/Datatang/Vehicle Image from Surveillance Video with Model Annotation"
cd "${folder}"
echo jpg: `find "${folder}" -name *.jpg | wc -l`

cd $PWD
for jpg in `find "${folder}" -name *.jpg`;do
    ./darknet detector test cfg/coco.data cfg/yolo.cfg ../weights/yolo.weights $jpg
done
