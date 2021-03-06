# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection1 import Detection1
from deep_sort.tracker import Tracker
from deep_sort.tracker1 import Tracker1
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture('top_view1.avi')
    video_capture_1 = cv2.VideoCapture('demo1.avi')

    # if writeVideo_flag:
    #     # Define the codec and create VideoWriter object
    #     w = int(video_capture.get(3))
    #     h = int(video_capture.get(4))
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     out = cv2.VideoWriter('output1.avi', fourcc, 15, (w, h))
    #     list_file = open('detection.txt', 'w')
    #     frame_index = -1

    fps = 0.0
    fig = plt.figure()
    fig1 = plt.figure()
    count = 0
    count1 = 0
    x_list = []
    y_list = []
    x_list1 = []
    y_list1 = []
    # ax1 = fig.add_subplot(1, 1, 1)
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        ret1, frame1 = video_capture_1.read()  # frame shape 640*480*3
        # if ret == True:
        #     print(' VIDEO FOUND')
       #  t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        image1 = Image.fromarray(frame1[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)
        boxs1 = yolo.detect_image(image1)
        print("box_co-ordinate = ", (boxs))
        print("box_co-ordinate = ", (boxs1))
        features = encoder(frame, boxs)
        features1 = encoder(frame1, boxs1)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        detections1 = [Detection1(bbox1, 1.0, feature1) for bbox1, feature1 in zip(boxs1, features1)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        boxes1 = np.array([d.tlwh for d in detections1])
        scores1 = np.array([d.confidence for d in detections1])
        indices1 = preprocessing.non_max_suppression(boxes1, nms_max_overlap, scores1)
        detections1 = [detections1[i] for i in indices1]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # tracker1.predict()
        # tracker1.update(detections1)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # for track1 in tracker1.tracks:
        #     if not track1.is_confirmed() or track1.time_since_update > 1:
        #         continue
        #     bbox1 = track1.to_tlbr()
        #     cv2.rectangle(frame1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), (255, 255, 255), 2)
        #     cv2.putText(frame1, str(track1.track_id), (int(bbox1[0]), int(bbox1[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:

            bbox = det.to_tlbr()

            # print((type(bbox)))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # print("The co-ordinates are:", int(bbox[0]), int(bbox[1]))
        # for det1 in detections1:

        #     bbox1 = det1.to_tlbr()

        #     # print((type(bbox)))
        #     cv2.rectangle(frame1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), (255, 0, 0), 2)
        #     # print("The co-ordinates are:", int(bbox[0]), int(bbox[1]))

        try:

            for i in boxs:
                x = (i[0] + i[2]) / 2
                y = (i[1] + i[3]) / 2
                count += 1
                x_list.append(x)
                y_list.append(y)
                if count == 1:
                    points = plt.scatter(x_list, y_list)
                elif count > 1:
                    print('x:', x_list, 'y:', y_list)
                    points.remove()
                    points = plt.scatter(x_list, y_list)
                    # plt.pause(0.9)
            x_list.clear()
            y_list.clear()
        except:
            continue

        try:

            for i in boxs1:
                x = (i[0] + i[2]) / 2
                y = (i[1] + i[3]) / 2
                count1 += 1
                x_list1.append(x)
                y_list1.append(y)
                if count1 == 1:
                    points = plt.scatter(x_list1, y_list1)
                elif count1 > 1:
                    print('x:', x_list1, 'y:', y_list1)
                    points.remove()
                    points = plt.scatter(x_list1, y_list1)
                    # plt.pause(0.9)
            x_list1.clear()
            y_list1.clear()

        except:
            continue

        #     # redraw the canvas
        fig.canvas.draw()
        fig1.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # for second frame

        img1 = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8,
                             sep='')
        img1 = img1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # display image with opencv or any operation you like
        cv2.imshow("plot", img)

        cv2.imshow('frame', frame)

        cv2.imshow("plot2", img1)

        cv2.imshow('frame1', frame1)

        # if writeVideo_flag:
        #     # save a frame
        #     out.write(frame)
        #     frame_index = frame_index + 1
        #     list_file.write(str(frame_index) + ' ')
        #     if len(boxs) != 0:
        #         for i in range(0, len(boxs)):
        #             list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
        #     list_file.write('\n')

        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %f" % (fps))

        # # Press Q to stop!
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    video_capture.release()
    # if writeVideo_flag:
    #     out.release()
    #     list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
