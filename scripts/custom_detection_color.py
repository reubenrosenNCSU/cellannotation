#!/usr/bin/env python
# coding: utf-8

import keras
import sys
import matplotlib.pyplot as plt
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

import cv2
import csv
import os
import numpy as np
from scipy.io import loadmat

# %% Functions

def listFile(path, ext):
    '''    
    Parameters
    ----------
    path : string
        Directory of processing images. 
    ext : string
        Desired file extension.

    Returns
    -------
    A list of all files with specific extension in a directory (including subdirectory).
    '''
    filename_list, filepath_list = [], []
    for r, d, f in os.walk(path):
        for filename in f:
            if ext in filename:
                filename_list.append(filename)
                filepath_list.append(os.path.join(r, filename))
    return sorted(filename_list), sorted(filepath_list)

def listTile(path):
    # Return a list of directories of tiles
    dir_list = []
    dirname_list = []
    for r, d, f in os.walk(path):
        if not d:
            dir_list.append(r)
            dirname_list.append(os.path.basename(r))
    return sorted(dirname_list), sorted(dir_list)


def non_max_suppression_merge(boxes, overlapThresh=0.5, sort=4):
    '''
    https://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
    '''
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
    ''' stitch predictions on a single tile image '''
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

# %% Main code

if __name__ == "__main__":
    # Command-line arguments
    image_path = sys.argv[1]  # Image file path
    h5_file = sys.argv[2]  # H5 file path
    output_folder = sys.argv[3]  # Output folder

    # Labels for detection
    labels_to_names = {0: 'uncertain', 1: 'yellow neuron', 2: 'yellow astrocyte', 
                        3: 'green neuron', 4: 'green astrocyte', 5: 'red neuron', 
                        6: 'red astrocyte'}

    tHRESHOLD = 0.5  # threshold for detection confidence score
    xsize = 512
    ysize = 512
    step = 448  # initial step size, can be adjusted dynamically based on image size

    classes = list(labels_to_names.values())
    num_class = len(classes)

    # load retinanet model
    model = models.load_model(h5_file, backbone_name='resnet50')
    model = models.convert_model(model)  # Convert to inference model

    # Initialize variables
    all_detections = []
    testnames = [os.path.basename(image_path)]
    testpaths = [image_path]
    CSV = os.path.join(output_folder, testnames[0] + '_result.csv')

    # Read and prepare image
    fullimg_c1 = read_image_bgr(image_path)
    fullimg = np.zeros(fullimg_c1.shape, dtype=np.uint16)
    fullimg[:, :, 2] = fullimg_c1[:, :, 2].copy()  # BGR
    fullimg[:, :, 1] = fullimg_c1[:, :, 1]

    if (fullimg[:, :, 2].sum() > 0) & (fullimg[:, :, 1].sum() > 0):
        fulldraw = fullimg.copy() / 257  # RGB to save
        fulldraw = (fulldraw * 3).clip(0, 255)  # Increase brightness

        # Padding for uneven dimensions
        H0, W0, _ = fullimg.shape
        step_x = min(xsize, W0)  # Ensure step_x does not exceed the width
        step_y = min(ysize, H0)  # Ensure step_y does not exceed the height

        # Adjust step size if image is smaller than tile size
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

        # Process patches
        for x in range(0, W, step_x):
            for y in range(0, H, step_y):
                offset = np.array([x, y, x, y])

                # Load image patch
                image = fullimg_pad[y:y + step_y, x:x + step_x]

                # Preprocess the image
                image = preprocess_image(image)
                image, scale = resize_image(image)

                # Process the image through the model
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

                # Adjust bounding boxes based on the image scale and offset
                boxes /= scale
                boxes += offset
                boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, W0)
                boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, H0)

                # Filter out detections below the threshold
                indices = np.where(scores[0, :] > tHRESHOLD)[0]

                # Sort the scores
                scores = scores[0][indices]
                scores_sort = np.argsort(-scores)

                # Save the detections
                image_boxes = boxes[0, indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[0, indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1
                )
                raw_detections = np.append(raw_detections, image_detections, axis=0)

        # Stitch detections and draw bounding boxes
        detections = stitchDetection(raw_detections, H, W, xsize=xsize, ysize=ysize, step=step_x)

        for detection in detections:
            box = detection[:4].astype(int)
            score = detection[4]
            label = int(detection[5])
            color = label_color(label)

            # Draw the bounding boxes on the image
            cv2.rectangle(fulldraw, (box[0], box[1]), (box[2], box[3]), color, 2)
            caption = f'{labels_to_names[label]} {score:.2f}'
            draw_caption(fulldraw, box, caption)

        # Save the processed image in the output folder
        output_image_path = os.path.join(output_folder, testnames[0] + '_detected.png')
        cv2.imwrite(output_image_path, fulldraw)

        # Output to CSV in the csv_output folder
        with open(CSV, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['xmin', 'ymin', 'xmax', 'ymax', 'score', 'class'])
            for detection in detections:
                writer.writerow([*detection[:-1], labels_to_names[int(detection[-1])]])

    else:
        print(f"Skipping image {testnames[0]} due to no valid data.")
