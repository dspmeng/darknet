#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#include <fenv.h>
#include <sys/stat.h>

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

char *barrier_names[] = {
    "minibus",
    "minitruck",
    "car",
    "mediumbus",
    "mpv",
    "suv",
    "largetruck",
    "largebus",
    "other"
};

char* detrac_names[] = {
    "car",
    "van",
    "bus",
    "others"
};

void train_yolo(char *cfgfile, char *weightfile, char *backup)
{
    char *train_images = "../pascal_voc/train.txt";
    char *backup_directory = "./backup-v1-tiny-yolo/";
    if (backup) backup_directory = backup;
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.c = net.c;
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile, char *results)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char base[1024];
    if (results) {
        snprintf(base, 1024, "%s/comp4_det_test_", results);
    } else {
        snprintf(base, 1024, "%s/comp4_det_test_", "results");
    }
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("../pascal_voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.c = net.c;
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, l.side*l.side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_detection_boxes(l, orig.w, orig.h, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

#ifdef OPENCV
void infer_video_yolo(network net, const char* input, float thresh, char *filename, char* results)
{
    image **alphabet = load_alphabet();
    float nms=.4;
    detection_layer l = net.layers[net.n-1];
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    int j;
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    CvCapture* capture = cvCaptureFromFile(input);
    IplImage* src = NULL;
    CvVideoWriter* writer = NULL;
    int count = 0;
    float total_time = 0;
    clock_t time;
    while (src = cvQueryFrame(capture)) {
        image im = ipl_to_image(src);
        CvSize size = {im.w, im.h};
        if (writer == NULL) {
            writer = cvCreateVideoWriter("out.avi", CV_FOURCC('X','2','6','4'), 25.0, size, 1);
        }

        rgbgr_image(im);
        image sized = resize_image(im, net.w, net.h);

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        total_time += sec(clock()-time);
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        bool has_obj = draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, barrier_names/*voc_names*/, alphabet, l.classes);

        if (writer && has_obj) {
            IplImage* dst = cvCreateImage(size, IPL_DEPTH_8U, 3);
            image_to_ipl(im, dst);
            cvWriteFrame(writer, dst);
            cvReleaseImage(&dst);
        }

        free_image(im);
        free_image(sized);
        printf("\r%d frames processed", count++);
        fflush(stdout);
    }

    printf("%s: %d frames, average %f seconds.\n", input, count, total_time / count);

    if (writer) cvReleaseVideoWriter(&writer);
    if (capture) cvReleaseCapture(&capture);
    free(boxes);
    free_ptrs((void **)probs, l.side*l.side*l.n);
    free_alphabet(alphabet);
}
#endif

void infer_image_yolo(network net, const char* input, float thresh, char *filename, char* results)
{
    image **alphabet = load_alphabet();
    float nms=.4;
    detection_layer l = net.layers[net.n-1];
    clock_t time;
    image im = load_image(input,0,0,net.c);
    image sized = resize_image(im, net.w, net.h);
    float *X = sized.data;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    char save_as[1024];
    int j;
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    time=clock();
    network_predict(net, X);
    printf("%s: Predicted with threshold %f in %f seconds.\n", input, thresh, sec(clock()-time));
//#define DUMP_DETECTION_LAYER
#ifdef DUMP_DETECTION_LAYER
    printf("detection layer [side %d, classes %d, n %d] has %d outputs:\n",
           l.side, l.classes, l.n, l.outputs);
    {
        int i, j;
        FILE* fp = fopen("detection_layer.txt", "w");
        if (fp) {
            float* output = l.output;
            for (i = 0; i < l.side * l.side; i++) {
                for (j = 0; j < l.classes; j++) {
                    fprintf(fp, "%9.6f ", output[i * l.classes + j]);
                }
                fprintf(fp, "\n");
            }
            output += l.side * l.side * l.classes;
            for (i = 0; i < l.side * l.side; i++) {
                for (j = 0; j < l.n; j++) {
                    fprintf(fp, "%9.6f ", output[i * l.n + j]);
                }
                fprintf(fp, "\n");
            }
            output += l.side * l.side * l.n;
            for (i = 0; i < l.side * l.side; i++) {
                for (j = 0; j < l.coords; j++) {
                    fprintf(fp, "%9.6f ", output[i * l.coords + j]);
                }
                fprintf(fp, "\n");
            }
        } else {
            fprintf(stderr, "Failed to open detection_layer.txt\n");
        }
        fclose(fp);
    }
    {
        FILE* fp = fopen("detection_layer.bin", "wb");
        fwrite(l.output, sizeof(float), l.outputs, fp);
        fclose(fp);
    }
#endif
    get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
    if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, alphabet, 20);
    printf("detection_thresh: %f, nms_thresh: %f\n", thresh, nms);
    draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, /*detrac_names*/barrier_names/*voc_names*/, alphabet, l.classes);
    if (results) {
        strncpy(save_as, results, strlen(results));
        save_as[strlen(results)] = '\0';
        strcat(save_as, &input[strlen(filename)]);
        printf("save as %s\n", save_as);
        save_image(im, save_as);
    } else {
        save_image(im, "predictions");
        show_image(im, "predictions");
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
    }

    free_image(im);
    free_image(sized);
    free(boxes);
    free_ptrs((void **)probs, l.side*l.side*l.n);
    free_alphabet(alphabet);
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, char *results, float thresh)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    char buff[256];
    char *input = buff;

    struct stat buf;
    FILE *fp = NULL;
    char cmdstr[256];
    if (filename && lstat(filename, &buf) < 0) {
        fprintf(stderr, "lstat error on %s\n", filename);
        return;
    }
    if (S_ISDIR(buf.st_mode)) {
        snprintf(cmdstr, 256, "ls -1 %s", filename);
        fp = popen(cmdstr, "r");
        if (fp == NULL) {
            fprintf(stderr, "Can't open %s\n", filename);
            return;
        }
        strncpy(input, filename, strlen(filename));
    }

    while(1){
        if (fp != NULL) {
            if (fgets(input + strlen(filename), 128, fp) == NULL)
                break;
            // remove new line from fgets
            input[strlen(input) - 1] = '\0';
        } else if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
#ifdef OPENCV
        int len = strlen(input);
        if (len > 4 &&
            (!strcmp(input + len - 4, ".avi") ||
             !strcmp(input + len - 4, ".mp4"))) {
            infer_video_yolo(net, input, thresh, filename, results);
        } else
#endif
        infer_image_yolo(net, input, thresh, filename, results);
        if (!S_ISDIR(buf.st_mode) && filename) break;
    }
}

void run_yolo(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    char *backup = find_char_arg(argc, argv, "-backup", 0);
    char *results = find_char_arg(argc, argv, "-results", 0);

    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    //feenableexcept(FE_INVALID | FE_OVERFLOW);

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, results, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights, backup);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights, results);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, 20, frame_skip, prefix, .5, 0,0,0,0);
}
