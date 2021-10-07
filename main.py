import numpy as np
import os
import csv
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance_matrix
import matplotlib.image as imgs
import matplotlib.pyplot as plt
from colors import get_colors

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def rgbs_to_to_hsvs(rgbs):
    hsv_s = np.zeros([3,3])
    hsv_s[0, :] = rgb_to_hsv(rgbs[0, 0], rgbs[0, 1], rgbs[0, 1])
    hsv_s[1, :] = rgb_to_hsv(rgbs[1, 0], rgbs[1, 1], rgbs[1, 1])
    hsv_s[2, :] = rgb_to_hsv(rgbs[2, 0], rgbs[2, 1], rgbs[2, 1])

    return hsv_s


kobayashi_colors_new = get_colors()


#for filename in os.listdir('/data/fnocentini/outfits/'):
for i in range(0, 170000):
    filename = str(i) + '.png'
    img = cv2.imread('/data/fnocentini/outfits/' + '100.png')
    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    img = result
    img = np.array(img).reshape((img.shape[1] * img.shape[0], 4))
    rows_to_delete = []
    for i, row in enumerate(img):
        if row[3] == 0:
            rows_to_delete.append(i)
    img2 = np.delete(img, rows_to_delete, axis=0)
    img3 = np.zeros(img2.shape)

    for i, row in enumerate(img2):
        img3[i, 0:3] = rgb_to_hsv(row[2], row[1], row[0]) #In bgr

    kmeans = KMeans(n_clusters=3, random_state=0).fit(img3[:, 0:3])
    clusters = kmeans.cluster_centers_.astype(int)

    #clusters[:, [0, 2]] = clusters[:, [2, 0]]  #Convert BGR to RGB
    #plt.imshow([clusters])
    #plt.show()

    if clusters.shape == (3, 3):
        distance = 10000
        for triplet in kobayashi_colors_new:
            current_distances = np.zeros(6)
#             current_distances[0] = np.mean(abs(clusters - (triplet.get_rgb())))
#             current_distances[1] = np.mean(abs(clusters - (triplet.get_rgb_1())))
#             current_distances[2] = np.mean(abs(clusters - (triplet.get_rgb_2())))
#             current_distances[3] = np.mean(abs(clusters - (triplet.get_rgb_3())))
#             current_distances[4] = np.mean(abs(clusters - (triplet.get_rgb_4())))
#             current_distances[5] = np.mean(abs(clusters - (triplet.get_rgb_5())))
            current_distances[0] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb()))))
            current_distances[1] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb_1()))))
            current_distances[2] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb_2()))))
            current_distances[3] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb_3()))))
            current_distances[4] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb_4()))))
            current_distances[5] = np.mean(abs(clusters - (rgbs_to_to_hsvs(triplet.get_rgb_5()))))


            if np.min(current_distances) < distance:
                best_triplet = triplet
                distance = np.min(current_distances)
                index = np.argmin(current_distances)
    if distance <= 10:
        with open('/data/fnocentini/good_outfits_subtraction_hsv.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([filename, distance, best_triplet.class_, best_triplet.subclass_])
        file.close()




