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
from deep_sort.tracker import Tracker
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

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output1.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    fig = plt.figure()
    count = 0
    x_list = []
    y_list = []
    x= []
    y=[]
    a1=[]
    c = []
    im=[]
    a1=[]
    # ax1 = fig.add_subplot(1, 1, 1)
    pts_src = np.array([[900,524],[1550,313],[650,236],[0,447]])

    pts_dst = np.array([[0, 0],[1000, 0],[1000, 1000],[0, 1000]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    # print('h=',h)

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            print('NO VIDEO FOUND')
            break
            
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)
        # print("box_co-ordinate = ", (boxs))

        # for i in boxs:
        #     print(i[0][0])

        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        for det in detections:

            bbox = det.to_tlbr()

            # print(((bbox)))
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # print("The co-ordinates are:", int(bbox[0]), int(bbox[1]))

        try:

            for i in boxs:
                x = (i[0] + i[2]) / 2
                y = (i[1] + i[3]) / 2
                count += 1
                x_list.append(x)
                y_list.append(y)
                print('x_list=', x_list, 'y_list=', y_list)
                z = zip(x_list,y_list)
                # print('z:',z)
                
                for i in z:
                    c.append(i)
                # print('c=',c)
                    a = np.array([c], dtype='float32')
                    a = np.array([c])
                    a1.append(a)

                    # # finally, get the mapping
                    imOut = cv2.perspectiveTransform(a1, h)
                    imOut = np.array([imOut])
                    # im.append(imOut)
                    print("imOut=",imOut)

                # for i in im:
                #     for j in i[0]:
                #         # count += 1

                #         x.append(j[0])
                #         y.append(j[1])


                #     # if count == 1:
                #         points = plt.scatter(x,y)
                #     # elif count > 1:
                        # points.remove()
                        # points = plt.scatter(x,y)
                            # plt.pause(0.9)
                    x_list.clear()
                    y_list.clear()
                    x.clear()
                    y.clear()
                    im.clear()
                    a1.clear()

        except:
            continue

            # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot", img)

        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
