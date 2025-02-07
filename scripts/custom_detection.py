#!/usr/bin/env python
# coding: utf-8

import keras
import sys
import os
import numpy as np
import cv2
import csv
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# %% Functions (Unchanged from original)
def listFile(path, ext):
    filename_list, filepath_list = [], []
    for r, d, f in os.walk(path):
        for filename in f:
            if ext in filename:
                filename_list.append(filename)
                filepath_list.append(os.path.join(r, filename))
    return sorted(filename_list), sorted(filepath_list)

def listTile(path):
    dir_list = []
    dirname_list = []
    for r, d, f in os.walk(path):
        if not d:
            dir_list.append(r)
            dirname_list.append(os.path.basename(r))
    return sorted(dirname_list), sorted(dir_list)

def non_max_suppression_merge(boxes, overlapThresh=0.5, sort=4):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    idxs = np.argsort(boxes[:, sort])

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        if np.where(overlap > overlapThresh)[0].size > 0:
            xxx1 = min(x1[i], x1[idxs[np.where(overlap > overlapThresh)[0]]].min())
            yyy1 = min(y1[i], y1[idxs[np.where(overlap > overlapThresh)[0]]].min())
            xxx2 = max(x2[i], x2[idxs[np.where(overlap > overlapThresh)[0]]].max())
            yyy2 = max(y2[i], y2[idxs[np.where(overlap > overlapThresh)[0]]].max())
            boxes[i, :4] = [xxx1, yyy1, xxx2, yyy2]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

def stitchDetection(detections, H, W, xsize=512, ysize=512, step=448):
    x_overlap = xsize - step
    y_overlap = ysize - step
    rows = []
    for row in range(step, W, step):
        rows.extend(list(range(row - 32, row + x_overlap + 32)))
    cols = []
    for col in range(step, H, step):
        cols.extend(list(range(col - 32, col + y_overlap + 32)))

    overlap_idx = []
    for i, detection in enumerate(list(detections)):
        box = list(map(int, detection[:-1]))
        if (box[0] in rows) or (box[1] in cols):
            overlap_idx.append(i)

    overlap_detections = detections[overlap_idx].copy()
    clean_mask = np.ones(detections.shape[0], dtype=bool)
    clean_mask[overlap_idx] = False
    clean_detections = detections[clean_mask].copy()

    if overlap_detections.size > 1:
        overlap_detections = non_max_suppression_merge(overlap_detections)
        clean_detections = np.append(clean_detections, overlap_detections, axis=0)

    return clean_detections

# %% Main code (Only modified input handling)
def detect_objects(image_path, model_path, output_folder, xsize=512, ysize=512, step=448, threshold=0.5):
    # Modified input handling
    if os.path.isdir(image_path):
        testnames, testpaths = listFile(image_path, '.tif')
    else:
        testnames = [os.path.basename(image_path)]
        testpaths = [image_path]

    # Original implementation below
    labels_to_names = {0: 'SGN'}
    tHRESHOLD = threshold
    classes = list(labels_to_names.values())
    num_class = len(classes)
    pATHRESULT = output_folder
    pATHCSV = output_folder

    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)

    all_detections = [[None for i in range(num_class)] for j in range(len(testnames))]
    clean_detections = [[None for i in range(num_class)] for j in range(len(testnames))]
    CSV = os.path.join(output_folder, 'custom_annotations.csv')

    for i, testpath in enumerate(testpaths):
        CSV = os.path.join(output_folder, 'custom_annotations.csv')
        with open(CSV, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', 
                                    quoting=csv.QUOTE_MINIMAL)
            
            fullimg_c1 = read_image_bgr(testpath)
            fullimg = np.zeros(fullimg_c1.shape, dtype=np.uint16)
            fullimg[:, :, 2] = fullimg_c1[:, :, 2].copy()
            fullimg[:, :, 1] = fullimg_c1[:, :, 1]

            if (fullimg[:, :, 2].sum() > 0) or (fullimg[:, :, 1].sum() > 0):
                fulldraw = fullimg.copy() / 257
                fulldraw = (fulldraw * 3).clip(0, 255)

                H0, W0, _ = fullimg.shape
                step_x = min(xsize, W0)
                step_y = min(ysize, H0)

                if W0 < xsize:
                    step_x = W0
                if H0 < ysize:
                    step_y = H0

                if not (H0 - ysize) % step_y == 0:
                    H = H0 - H0 % step_y + ysize
                else:
                    H = H0
                if not (W0 - xsize) % step_x == 0:
                    W = W0 - W0 % step_x + xsize
                else:
                    W = W0

                if W != W0 or H != H0:
                    fullimg_pad = np.zeros((H, W, 3), dtype=np.uint16)
                    fullimg_pad[0:H0, 0:W0] = fullimg.copy()
                else:
                    fullimg_pad = fullimg.copy()

                n = 0
                raw_detections = np.empty((0, 6))

                for x in range(0, W, step_x):
                    for y in range(0, H, step_y):
                        offset = np.array([x, y, x, y])
                        image = fullimg_pad[y:y + step_y, x:x + step_x]
                        image = preprocess_image(image)
                        image, scale = resize_image(image)
                        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                        boxes /= scale
                        boxes += offset
                        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, W0)
                        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, H0)
                        indices = np.where(scores[0, :] > tHRESHOLD)[0]
                        scores = scores[0][indices]
                        scores_sort = np.argsort(-scores)
                        image_boxes = boxes[0, indices[scores_sort], :]
                        image_scores = scores[scores_sort]
                        image_labels = labels[0, indices[scores_sort]]
                        image_detections = np.concatenate(
                            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1
                        )
                        raw_detections = np.append(raw_detections, image_detections, axis=0)
            else:
                raw_detections = np.empty((0, 6))
            
            for label in range(num_class):
                all_detections[i][label] = raw_detections[raw_detections[:, -1] == label, :-1]
                detections = raw_detections[raw_detections[:, -1] == label, :-1].copy()
                if detections.size > 1:
                    cleaned_detections = stitchDetection(detections, H0, W0, xsize, ysize, step=step_x)
                else:
                    cleaned_detections = detections.copy()
                cleaned_detections = np.concatenate([cleaned_detections,
                                                            np.zeros([cleaned_detections.shape[0],1]),
                                                            np.ones([cleaned_detections.shape[0],1])*(i+1)],
                                                            axis=1)
                clean_detections[i][label] = cleaned_detections

                if cleaned_detections.size > 1:
                    for j, detection in enumerate(list(cleaned_detections)):
                        b = list(map(int, detection[:4]))
                        color = label_color(label)
                        draw_box(fulldraw, b, color=color, thickness=2)
                        cleaned_detections[j,5] = fullimg[b[1]:b[3],b[0]:b[2]].mean()
                        filewriter.writerow([testnames[i],
                                                        detection[0],detection[1],
                                                        detection[2],detection[3],
                                                        classes[label],detection[-3],
                                                        detection[-2],detection[-1]])
                            
                output_image_path = os.path.join(pATHRESULT, 'custom_detection.png')
                cv2.imwrite(output_image_path, cv2.cvtColor(fulldraw.astype('uint8'), cv2.COLOR_RGB2GRAY))

if __name__ == "__main__":
    image_path = sys.argv[1]
    h5file_path = sys.argv[2]
    output_folder = sys.argv[3]
    detect_objects(image_path, h5file_path, output_folder)