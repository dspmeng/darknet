#!/bin/bash

#set -x

usage ()
{
   echo "Error @line" $1
   echo "Usage: $0 [-t] -v YOLO_VERSION [-w WEIGHTS] [-i IMG] [-r THRESHOLD] train/test"
   echo "  -t    tiny YOLO or not"
   echo "  -v    YOLO version 1 or 2"
   echo "  -w    weight file"
   echo "  -i    test image"
   echo "  -r    recall threshold"
   exit 1
}

IMG="f0050_tiny.jpg"

while getopts ":tv:w:i:r:" opt; do
    case "${opt}" in
        t)
            TINY=true
            ;;
        v)
            VER=$OPTARG
            ;;
        w)
            WEIGHTS=$OPTARG
            ;;
        i)
            IMG=$OPTARG
            ;;
        r)
            THRESHOLD="-thresh $OPTARG"
            ;;
        *)
            ;;
    esac
done

shift $(($OPTIND - 1))

if [ -z $VER ] || [ $# -ne 1 ]; then
    usage ${LINENO}
fi

OP=$1

if [ $VER -eq 1 ]; then
    if [ $TINY ]; then
        echo "TINY YOLO v1"
        case "${OP}" in
            train)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="../weights/darknet.conv.weights"
                fi
                ./darknet yolo train cfg/yolov1/tiny-yolo.cfg $WEIGHTS
                ;;
            test)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="backup/tiny-yolo_final.weights"
                fi
                ./darknet yolo test cfg/yolov1/tiny-yolo.cfg $WEIGHTS $IMG $THRESHOLD -gpus 1
                ;;
            *)
                usage
                ;;
        esac
    else
        echo "Sorry, only TINY YOLO v1 supported."
        exit 1
    fi
elif [ $VER -eq 2 ]; then
    if [ $TINY ]; then
        echo "TINY YOLO v2"
        case "${OP}" in
            train)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="../weights/tiny-yolo-voc.weights"
                fi
                ./darknet detector train cfg/voc.data cfg/tiny-yolo-voc.cfg $WEIGHTS
                ;;
            test)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="backup/tiny-yolo-voc_final.weights"
                fi
                ./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg $WEIGHTS $IMG $THRESHOLD -gpus 1
                ;;
            *)
                usage
                ;;
        esac
    else
        echo "YOLO v2"
        case "${OP}" in
            train)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="../weights/darknet19_448.conv.23"
                fi
                ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg $WEIGHTS
                ;;
            test)
                if [ -z $WEIGHTS ]; then
                    WEIGHTS="backup/yolo-voc_final.weights"
                fi
                ./darknet detector test cfg/voc.data cfg/yolo-voc.cfg $WEIGHTS $IMG $THRESHOLD -gpus 1
                ;;
            *)
                usage
                ;;
        esac
    fi
else
    echo "Invalid YOLO version. Only support 1 or 2."
    exit 1
fi
