./darknet combine cfg/tiny-yolo-dss_barrier_all_v1-nobn.cfg ../weights/tiny-yolo-dss_barrier_all_v1_40000-nobn.weights tiny-yolo-dss_barrier_all_v1_40000-combined.weights
./darknet yolo valid cfg/tiny-yolo-dss_barrier_all_v1-combined.cfg tiny-yolo-dss_barrier_all_v1_40000-combined.weights -results results -testlist ../dss_barrier/test_all.txt
python scripts/run_ap.py -a ../dss_barrier/labels/ -i ../dss_barrier/test_all.txt -n ../dss_barrier/class_names.txt -d
