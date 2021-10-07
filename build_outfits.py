import csv
import cv2

with open('/data/fnocentini/data/train_clean.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    i = 0
    for row in csv_reader:
        #img1 = cv2.imread(row[0])
        #img2 = cv2.imread(row[1])
        #outfit = cv2.hconcat([img1, img2])

        #cv2.imwrite('/data/fnocentini/val_outfits/' + str(i) + '.png', outfit)
        i = i+1
    print(i)
