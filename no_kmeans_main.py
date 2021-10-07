import cv2
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from colors import get_colors
import csv

kobayashi_colors_new = get_colors()


rgb_CIS = np.array([[[231, 47, 39], [238, 113, 25], [255, 200, 8], [170, 198, 27], [19, 166, 50], [4, 148, 87], [1, 134, 141], [3, 86, 155], [46, 20, 141], [204, 63, 69]],
                        [[207, 46, 49], [226, 132, 45], [227, 189, 28], [162, 179, 36], [18, 154, 47], [6, 134, 84], [3, 130, 122], [6, 113, 148], [92, 104, 163], [175, 92, 87]],
                        [[231, 108, 86], [241, 176, 102], [255, 228, 15], [169, 199, 35], [88, 171, 45], [43, 151, 89], [0, 147, 159], [59, 130, 57], [178, 137, 166], [209, 100, 109]],
                        [[233, 163, 144], [242, 178, 103], [255, 236, 79], [219, 220, 93], [155, 196, 113], [146, 198, 131], [216, 188, 209], [147, 184, 213], [197, 188, 213], [218, 176, 176]],
                        [[236, 217, 202], [245, 223, 181], [249, 239, 189], [228, 235, 191], [221, 232, 207], [209, 234, 211], [194, 222, 242], [203, 215, 232], [224, 218,230], [235, 219, 224]],
                        [[213, 182, 166], [218, 196, 148], [233, 227, 143], [209, 116, 73], [179, 202, 157], [166, 201, 163], [127, 175, 166], [165, 184, 199], [184, 190, 189], [206, 185, 179]],
                        [[211, 142, 110], [215, 145, 96], [255, 203, 88], [195, 202, 101], [141, 188, 90], [140, 195, 110], [117, 173, 169], [138, 166, 187], [170, 165, 199], [205, 154, 149]],
                        [[171, 131, 115], [158, 128, 110], [148, 133, 105], [144, 135, 96], [143, 162, 121], [122, 165, 123], [130, 154, 145], [133, 154, 153], [151, 150, 139], [160, 147, 131]],
                        [[162, 88, 61], [167, 100, 67], [139, 117, 65], [109, 116, 73], [88, 126, 61], [39, 122, 62], [24, 89, 63], [53, 108, 98], [44, 77, 143], [115, 71, 79]],
                        [[172, 36, 48], [169, 87, 49], [156, 137, 37], [91, 132, 47], [20, 114, 48], [23, 106, 43], [20, 88, 60], [8, 87, 107], [58, 55, 109], [116, 61, 56]],
                        [[116, 47, 50], [115, 63, 44], [103, 91, 44], [54, 88, 48], [30, 98, 50], [27, 86, 49], [18, 83, 65], [16, 76, 84], [40, 57, 103], [88, 60, 50]],
                        [[79, 46, 43], [85, 55, 43], [75, 63, 45], [44, 60, 49], [34, 62, 51], [31, 56, 45], [29, 60, 67], [25, 62, 63], [34, 54, 68], [53, 52, 48]],
                        [[10, 10, 10], [38, 38, 38], [60, 60, 60], [86, 86, 86], [126, 126, 126], [152, 152, 152], [180, 180, 180], [206, 206, 206], [236, 236, 236], [244, 244, 244]]])
rgb_CIS = np.array(rgb_CIS).reshape((rgb_CIS.shape[1] * rgb_CIS.shape[0], 3))
for i in range(25000):
    filename = str(i) + '.png'
    img_o = cv2.imread('/data/fnocentini/val_outfits/' + filename)
    img_a = img_o
    # convert to graky
    gray = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img_o.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    img = result
    img = np.array(img).reshape((img.shape[1] * img.shape[0], 4))
    rows_to_delete = []
    for i, row in enumerate(img):
        if row[3] == 0:
            rows_to_delete.append(i)
    img = np.delete(img, rows_to_delete, axis=0)
    img = img[:, 0:3] #Convert BGRA to BGR
    img[:, [0, 2]] = img[:, [2, 0]]  # Convert BGR to RGB
    # print(img.shape)
    img_o = np.array(img_o).reshape((img_o.shape[1] * img_o.shape[0], 3))
    d = distance_matrix(img, rgb_CIS)
    d1 = distance_matrix(img_o, rgb_CIS)
    #d = euclidean_distances(img, rgb_CIS)
    indices = d.argmin(axis=1)
    #indices1 = d1.argmin(axis=1)
    img = rgb_CIS[indices]
    #img_o = rgb_CIS[indices1]
    histogram, _ = np.histogram(indices, bins=range(130))
    indices = (-histogram).argsort()[0:3]
    clusters = rgb_CIS[indices]
    # plt.imshow([clusters])
    # plt.savefig('/data/fnocentini/cluster.png')
    #plt.show()
    #img_o = img_o.reshape((150, 300, 3))
    #cv2.imwrite('data/fnocentini/prova.png', img_o)
    if clusters.shape == (3, 3):
        distance = 10000
        for triplet in kobayashi_colors_new:
            current_distances = np.zeros(6)
            current_distances[0] = np.mean(abs(clusters - (triplet.get_rgb())))
            current_distances[1] = np.mean(abs(clusters - (triplet.get_rgb_1())))
            current_distances[2] = np.mean(abs(clusters - (triplet.get_rgb_2())))
            current_distances[3] = np.mean(abs(clusters - (triplet.get_rgb_3())))
            current_distances[4] = np.mean(abs(clusters - (triplet.get_rgb_4())))
            current_distances[5] = np.mean(abs(clusters - (triplet.get_rgb_5())))

            if np.min(current_distances) < distance:
                best_triplet = triplet
                distance = np.min(current_distances)
                index = np.argmin(current_distances)

    if distance <= 10:
         with open('/data/fnocentini/good_outfits_no_kmeans_val.csv', 'a') as file:
             csv_writer = csv.writer(file)
             csv_writer.writerow([filename, distance, best_triplet.class_, best_triplet.subclass_])
         file.close()



