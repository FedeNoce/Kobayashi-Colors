import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv

imgs_path = '/data/fnocentini/val_outfits/'
train_imgs_name = []
train_labels = []
labels_ = ['ROMANTIC', 'CLEAR', 'COOL-CASUAL', 'NATURAL', 'ELEGANT', 'FORMAL', 'MODERN', 'CHIC', 'DANDY', 'CLASSIC', 'GORGEOUS', \
           'ETHNIC', 'DYNAMIC', 'CASUAL', 'PRETTY']

for label in labels_:
     os.mkdir('/data/fnocentini/test_dataset/' + label)
with open('/data/fnocentini/data/good_outfits_no_kmeans_val.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if row[2] != 'NO CLASS DEFINED':
            img = cv2.imread(imgs_path + row[0])
            cv2.imwrite('/data/fnocentini/test_dataset/' + str(row[2]) + '/' + str(row[0]), img)
            train_imgs_name.append(row[0])
            train_labels.append(row[2])

