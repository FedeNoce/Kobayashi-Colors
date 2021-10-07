import numpy as np
import numpy as np
import os
import csv
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance_matrix
import matplotlib.image as imgs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import whiten, kmeans
import matplotlib.pyplot as plt


class Kobayashi_color():
    def __init__(self, rgb, subclass_, class_=''):
        self.rgb = rgb
        self.class_ = ''
        self.subclass_ = subclass_
        if self.subclass_ == 'Ethnic' or \
           self.subclass_ == 'Untamed' or \
           self.subclass_ == 'Fruitful' or \
           self.subclass_ == 'Wild' or \
           self.subclass_ == 'Robust':
            self.class_ = 'ETHNIC'
        if self.subclass_ == 'Authoritative':
            self.class_ = 'FORMAL'
        if self.subclass_ == 'Grand' or \
           self.subclass_ == 'Intrepid' or \
           self.subclass_ == 'Salty' or \
           self.subclass_ == 'Aqueous' or \
           self.subclass_ == 'Pastoral' or \
           self.subclass_ == 'Artistic' or \
           self.subclass_ == 'Deep' or \
           self.subclass_ == 'Dewy':
            self.class_ = 'NO CLASS DEFINED'
        if self.subclass_ == 'Charming' or \
           self.subclass_ == 'Agreeable to the touch' or \
           self.subclass_ == 'Soft' or \
           self.subclass_ == 'Sweet & Dreamy' or \
           self.subclass_ == 'Amiable' or \
           self.subclass_ == 'Supple' or \
           self.subclass_ == 'Dreamy' or \
           self.subclass_ == 'Innocent' or \
           self.subclass_ == 'Romantic':
            self.class_ = 'ROMANTIC'
        if self.subclass_ == 'Light' or \
           self.subclass_ == 'Neat' or \
           self.subclass_ == 'Fresh & Young' or \
           self.subclass_ == 'Clear' or \
           self.subclass_ == 'Pure' or \
           self.subclass_ == 'Clean' or \
           self.subclass_ == 'Crystalline' or \
           self.subclass_ == 'Refreshing' or \
           self.subclass_ == 'Simple' or \
           self.subclass_ == 'Pure & Simple' or \
           self.subclass_ == 'Clean & Fresh':
            self.class_ = 'CLEAR'
        if self.subclass_ == 'Youthful' or \
           self.subclass_ == 'Steady' or \
           self.subclass_ == 'Young' or \
           self.subclass_ == 'Speedy' or \
           self.subclass_ == 'Agile' or \
           self.subclass_ == 'Western' or \
           self.subclass_ == 'Sporty' or \
           self.subclass_ == 'Smart' or \
           self.subclass_ == 'Polished':
            self.class_ = 'COOL-CASUAL'
        if self.subclass_ == 'Urbane' or \
           self.subclass_ == 'Composite' or \
           self.subclass_ == 'Progressive' or \
           self.subclass_ == 'Distinguished' or \
           self.subclass_ == 'Intellectual' or \
           self.subclass_ == 'Modern' or \
           self.subclass_ == 'Cultivated' or \
           self.subclass_ == 'Precise' or \
           self.subclass_ == 'Exact' or \
           self.subclass_ == 'Rational' or \
           self.subclass_ == 'Metallic' or \
           self.subclass_ == 'Sublime' or \
           self.subclass_ == 'Earnest' or \
           self.subclass_ == 'Proper' or \
           self.subclass_ == 'Composed' or \
           self.subclass_ == 'Masculine':
            self.class_ = 'MODERN'
        if self.subclass_ == 'Diligent' or \
           self.subclass_ == 'Subtle & Mysterious' or \
           self.subclass_ == 'Quiet & Sophisticated' or \
           self.subclass_ == 'Eminent' or \
           self.subclass_ == 'Bitter' or \
           self.subclass_ == 'Placid' or \
           self.subclass_ == 'Aristocratic' or \
           self.subclass_ == 'Dapper' or \
           self.subclass_ == 'Precious' or \
           self.subclass_ == 'Formal' or \
           self.subclass_ == 'Solemn' or \
           self.subclass_ == 'Pratical' or \
           self.subclass_ == 'Sound' or \
           self.subclass_ == 'Majestic' or \
           self.subclass_ == 'Heavy & Deep' or \
           self.subclass_ == 'Strong & Robust' or \
           self.subclass_ == 'Serious' or \
           self.subclass_ == 'Dignified':
            self.class_ = 'DANDY'
        if self.subclass_ == 'Quiet' or \
           self.subclass_ == 'Chic' or \
           self.subclass_ == 'Noble & Elegant' or \
           self.subclass_ == 'Japanese' or \
           self.subclass_ == 'Modest' or \
           self.subclass_ == 'Simple, quiet & elegant' or \
           self.subclass_ == 'Sober' or \
           self.subclass_ == 'Adult' or \
           self.subclass_ == 'Stylish':
            self.class_ = 'CHIC'
        if self.subclass_ == 'Provincial' or \
           self.subclass_ == 'Rustic' or \
           self.subclass_ == 'Tasteful' or \
           self.subclass_ == 'Complex' or \
           self.subclass_ == 'Mellow' or \
           self.subclass_ == 'Old-fashioned' or \
           self.subclass_ == 'Classic' or \
           self.subclass_ == 'Traditional' or \
           self.subclass_ == 'Conservative' or \
           self.subclass_ == 'Elaborate' or \
           self.subclass_ == 'Splendid' or \
           self.subclass_ == 'Heavy' or \
           self.subclass_ == 'Sturdy':
            self.class_ = 'CLASSIC'
        if self.subclass_ == 'Feminine' or \
           self.subclass_ == 'Cultured' or \
           self.subclass_ == 'Delicate' or \
           self.subclass_ == 'Tender' or \
           self.subclass_ == 'Natural' or \
           self.subclass_ == 'Emotional' or \
           self.subclass_ == 'Dry' or \
           self.subclass_ == 'Simple & Appealing' or \
           self.subclass_ == 'Sleek' or \
           self.subclass_ == 'Pure & Elegant' or \
           self.subclass_ == 'Sedate' or \
           self.subclass_ == 'Noble' or \
           self.subclass_ == 'Fashionable' or \
           self.subclass_ == 'Refined' or \
           self.subclass_ == 'Subtle' or \
           self.subclass_ == 'Interesting' or \
           self.subclass_ == 'Mysterious' or \
           self.subclass_ == 'Graceful' or \
           self.subclass_ == 'Elegant' or \
           self.subclass_ == 'Gentle & Elegant' or \
           self.subclass_ == 'Brilliant' or \
           self.subclass_ == 'Genteel' or \
           self.subclass_ == 'Calm':
            self.class_ = 'ELEGANT'
        if self.subclass_ == 'Nostalgic' or \
           self.subclass_ == 'Delicious' or \
           self.subclass_ == 'Mild' or \
           self.subclass_ == 'Open' or \
           self.subclass_ == 'Domestic' or \
           self.subclass_ == 'Smooth' or \
           self.subclass_ == 'Healty' or \
           self.subclass_ == 'Restful' or \
           self.subclass_ == 'Sweet-sour' or \
           self.subclass_ == 'Free' or \
           self.subclass_ == 'Pleasant' or \
           self.subclass_ == 'Generous' or \
           self.subclass_ == 'Intimate' or \
           self.subclass_ == 'Gentle' or \
           self.subclass_ == 'Sunny' or \
           self.subclass_ == 'Wholesome' or \
           self.subclass_ == 'Citrus' or \
           self.subclass_ == 'Peaceful' or \
           self.subclass_ == 'Tranquil' or \
           self.subclass_ == 'Fresh' or \
           self.subclass_ == 'Plain' or \
           self.subclass_ == 'Friendly' or \
           self.subclass_ == 'Lighthearted':
            self.class_ = 'NATURAL'
        if self.subclass_ == 'Fascinating' or \
           self.subclass_ == 'Substantial' or \
           self.subclass_ == 'Glossy' or \
           self.subclass_ == 'Alluring' or \
           self.subclass_ == 'Aromatic' or \
           self.subclass_ == 'Mature' or \
           self.subclass_ == 'Extravagant' or \
           self.subclass_ == 'Gorgeous' or \
           self.subclass_ == 'Luxurious' or \
           self.subclass_ == 'Abundant' or \
           self.subclass_ == 'Decorative':
            self.class_ = 'GORGEOUS'
        if self.subclass_ == 'Lively' or \
           self.subclass_ == 'Hot' or \
           self.subclass_ == 'Provocative' or \
           self.subclass_ == 'Vigorous' or \
           self.subclass_ == 'Dynamic' or \
           self.subclass_ == 'Forceful' or \
           self.subclass_ == 'Bold' or \
           self.subclass_ == 'Dynamic & Active' or \
           self.subclass_ == 'Active' or \
           self.subclass_ == 'Fiery' or \
           self.subclass_ == 'Striking' or \
           self.subclass_ == 'Rich' or \
           self.subclass_ == 'Intense':
            self.class_ = 'DYNAMIC'
        if self.subclass_ == 'Cheerful' or \
           self.subclass_ == 'Happy' or \
           self.subclass_ == 'Enjoyable' or \
           self.subclass_ == 'Festive' or \
           self.subclass_ == 'Bright' or \
           self.subclass_ == 'Dazzling' or \
           self.subclass_ == 'Merry' or \
           self.subclass_ == 'Amusing' or \
           self.subclass_ == 'Casual' or \
           self.subclass_ == 'Flamboyant' or \
           self.subclass_ == 'Showy' or \
           self.subclass_ == 'Vivid' or \
           self.subclass_ == 'Tropical' or \
           self.subclass_ == 'Colorful':
            self.class_ = 'CASUAl'
        if self.subclass_ == 'Pretty' or \
           self.subclass_ == 'Cute' or \
           self.subclass_ == 'Childlike' or \
           self.subclass_ == 'Sweet':
            self.class_ = 'PRETTY'


    def get_rgb(self):
        #print(self.rgb)
        return self.rgb

    def get_rgb_1(self):
        tmp = np.zeros([3, 3])
        tmp[0, :] = self.rgb[2, :]
        tmp[1, :] = self.rgb[1, :]
        tmp[2, :] = self.rgb[0, :]
        #print(tmp)
        return tmp

    def get_rgb_2(self):
        tmp = np.zeros([3, 3])
        tmp[0, :] = self.rgb[1, :]
        tmp[1, :] = self.rgb[0, :]
        tmp[2, :] = self.rgb[2, :]
        #print(tmp)
        return tmp

    def get_rgb_3(self):
        tmp = np.zeros([3, 3])
        tmp[0, :] = self.rgb[1, :]
        tmp[1, :] = self.rgb[2, :]
        tmp[2, :] = self.rgb[0, :]
        #print(tmp)
        return tmp

    def get_rgb_4(self):
        tmp = np.zeros([3, 3])
        tmp[0, :] = self.rgb[0, :]
        tmp[1, :] = self.rgb[2, :]
        tmp[2, :] = self.rgb[1, :]
        #print(tmp)
        return tmp

    def get_rgb_5(self):
        tmp = np.zeros([3, 3])
        tmp[0, :] = self.rgb[2, :]
        tmp[1, :] = self.rgb[0, :]
        tmp[2, :] = self.rgb[1, :]
        #print(tmp)
        return tmp




    def set_rgb(self, new):
        self.rgb = new

def get_colors():
    kobayashi_colors = []


    kobayashi_colors.append(Kobayashi_color(np.array([[255,178,61],[254,245,249],[250,162,136]]), 'Sweet', 'Pretty'))
    kobayashi_colors.append(Kobayashi_color(np.array([[248,238,70],[253,247,177],[189,29,66]]), 'Pretty', 'Pretty'))
    kobayashi_colors.append(Kobayashi_color(np.array([[226,52,48],[253,248,191],[97,183,108]]), 'Childlike', 'Pretty'))
    kobayashi_colors.append(Kobayashi_color(np.array([[227,52,47],[247,236,69],[0,99,54]]), 'Enjoyable', 'Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[254,146,52],[254,247,251],[181,31,42]]), 'Bright', 'Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[181,31,40],[255,248,252],[34,45,113]]), 'Casual', 'Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[248,237,69],[238,77,48],[180,29,60]]), 'Colorful', 'Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[255,233,190],[255,248,252],[240,239,170]]), 'Soft', 'Romantic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[252,221,202],[253,186,178],[199,236,248]]), 'Charming', 'Romantic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[253,236,218],[255,248,250],[203,233,243]]), 'Dreamy', 'Romantic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[225,52,46],[34,44,112],[183,31,42]]), 'Lively', 'Dynamic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[255,211,55],[181,30,40],[5,6,8]]), 'Bold', 'Dynamic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[4,5,7],[184,30,41],[34,35,102]]), 'Active', 'Dynamic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[181,30,40],[5,6,10],[136,119,43]]), 'Wild', 'Dynamic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[208,120,117],[148,23,40],[16,14,66]]), 'Alluring', 'Gorgeous'))
    kobayashi_colors.append(Kobayashi_color(np.array([[143,24,29],[223,169,58],[17,15,66]]), 'Extravagant', 'Gorgeous'))
    kobayashi_colors.append(Kobayashi_color(np.array([[3,4,6],[181,35,42],[37,35,100]]), 'Mellow', 'Gorgeous'))
    kobayashi_colors.append(Kobayashi_color(np.array([[58,5,2],[5,6,11],[222,168,58]]), 'Luxurious', 'Gorgeous'))
    kobayashi_colors.append(Kobayashi_color(np.array([[149,70,29],[5,6,8],[79,94,47]]), 'Traditional', 'Classic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[34,12,11],[117,105,49],[10,18,29]]), 'Elaborate', 'Classic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[5,6,8],[138,45,30],[5,6,8]]), 'Heavy and deep', 'Classic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[103,84,37],[5,6,8],[140,138,122]]), 'Quiet and sophisticated', 'Dandy'))
    kobayashi_colors.append(Kobayashi_color(np.array([[41,44,49],[120,117,52],[4,5,7]]), 'Dapper', 'Dandy'))
    kobayashi_colors.append(Kobayashi_color(np.array([[5,6,7],[86,87,72],[5,6,8]]), 'Dignified', 'Dandy'))
    kobayashi_colors.append(Kobayashi_color(np.array([[5,6,7],[249,248,253],[0,100,54]]), 'Sharp', 'Modern'))
    kobayashi_colors.append(Kobayashi_color(np.array([[112,104,86],[255,248,252],[4,5,7]]), 'Rational', 'Modern'))
    kobayashi_colors.append(Kobayashi_color(np.array([[157,182,217],[0,61,99],[6,7,9]]), 'Masculine', 'Modern'))
    kobayashi_colors.append(Kobayashi_color(np.array([[6,6,6],[106,117,125],[4,5,7]]), 'Metallic', 'Modern'))
    kobayashi_colors.append(Kobayashi_color(np.array([[203,202,185],[99,142,90],[41,44,50]]), 'Chic', 'Chic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[138,120,110],[203,202,184],[102,124,170]]), 'Noble and Elegant', 'Chic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[29,35,101],[95,98,79],[203,202,185]]), 'Stylish', 'Chic'))
    kobayashi_colors.append(Kobayashi_color(np.array([[247,237,69],[94,178,86],[75,90,128]]), 'Youthful', 'Cool Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[0,100,54],[255,248,252],[0,103,143]]), 'Agile', 'Cool Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[255,212,56],[249,248,253],[34,44,112]]), 'Speedy', 'Cool Casual'))
    kobayashi_colors.append(Kobayashi_color(np.array([[253,220,201],[219,183,177],[110,139,182]]), 'Delicate', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[243,205,159],[191,168,198],[143,142,138]]), 'Emotional', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[170,168,154],[225,202,198],[112,135,121]]), 'Fashionable', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[139,120,112],[210,137,129],[201,188,201]]), 'Graceful', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[242,206,158],[147,116,72],[181,171,166]]), 'Calm', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[105,82,36],[146,117,72],[170,168,154]]), 'Modest', 'Elegant'))
    kobayashi_colors.append(Kobayashi_color(np.array([[255,233,189],[220,131,62],[147,116,73]]), 'Mild', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[253,233,170],[180,197,66],[147,138,80]]), 'Wholesome', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[239,157,55],[251,246,178],[249,240,126]]), 'Intimate', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[251,215,127],[255,233,204],[160,204,123]]), 'Tranquil', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[253,248,191],[208,220,119],[170,169,153]]), 'Plain', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[206,220,66],[123,192,76],[208,230,186]]), 'Fresh', 'Natural'))
    kobayashi_colors.append(Kobayashi_color(np.array([[122,191,76],[255,248,250],[72,107,146]]), 'Refreshing', 'Clear'))
    kobayashi_colors.append(Kobayashi_color(np.array([[125,187,229],[251,246,250],[146,171,208]]), 'Clean', 'Clear'))
    kobayashi_colors.append(Kobayashi_color(np.array([[204,230,204],[255,248,250],[207,236,245]]), 'Neat', 'Clear'))

    kobayashi_colors_new = []

    #R/V Carmine
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [244, 244, 244], [241, 176, 172]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [244, 244, 244], [255, 200, 8]]), 'Festive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [244, 244, 244], [3, 86, 155]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [139, 117, 65], [255, 200, 8]]), 'Hot'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [8, 87, 107], [255, 200, 8]]), 'Vigorous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [244, 244, 244], [10, 10, 10]]), 'Bold'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [10, 10, 10], [207, 46, 49]]), 'Forceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [10, 10, 10], [255, 200, 8]]), 'Dynamic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 47, 39], [4, 148, 87], [10, 10, 10]]), 'Dynamic & Active'))

    #R/S Rouge Coral
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [226, 132, 45], [255, 200, 8]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [233, 163, 144], [115, 71, 79]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [205, 154, 149], [115, 71, 79]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [169, 87, 49], [115, 63, 44]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [255, 200, 8], [178, 137, 166]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [44, 77, 143], [170, 165, 199]]), 'Fascinating'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [226, 132, 45], [88, 126, 61]]), 'Abundant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [255, 200, 8], [53, 52, 48]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[207, 46, 49], [115, 63, 44], [40, 57, 103]]), 'Mellow'))

    #R/P Rose
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [236, 217, 202], [242, 178, 103]]), 'Sweet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [255, 236, 79], [242, 178, 103]]), 'Cheerful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [249, 239, 189], [126, 188, 209]]), 'Childlike'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [255, 236, 79], [209, 100, 109]]), 'Joyful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [244, 244, 244], [255, 200, 8]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [255, 236, 79], [88, 171, 45]]), 'Merry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [175, 92, 87], [92, 104, 163]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [255, 228, 202], [178, 137, 166]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[231, 108, 66], [231, 47, 39], [244, 244, 244]]), 'Festive'))

    #R/P Flamingo
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [249, 239, 189], [242, 178, 103]]), 'Sweet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [255, 236, 79], [245, 223, 181]]), 'Preety'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [245, 223, 181], [213, 182, 166]]), 'Sunny'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [211, 142, 110], [242, 239, 189]]), 'Generous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [255, 236, 79], [155, 169, 113]]), 'Childlike'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [245, 223, 181], [218, 176, 176]]), 'Sweet & Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [219, 220, 93], [255, 236, 79]]), 'Sweet-sour'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [255, 236, 79], [126, 188, 209]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[233, 163, 144], [245, 223, 181], [184, 190, 189]]), 'Feminine'))

    #R/Vp Baby Pink
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [213, 182, 166], [233, 163, 144]]), 'Agreeable to the touch'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [218, 196, 148], [233, 227, 143]]), 'Amiable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [244, 244, 244], [221, 232, 207]]), 'Innocent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [213, 182, 166], [211, 142, 110]]), 'Smooth'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [184, 190, 189], [233, 227, 143]]), 'Gentle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [206, 206, 206], [221, 232, 207]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [218, 176, 176], [197, 188, 213]]), 'Sweet & Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [218, 176, 176], [224, 218, 230]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236, 217, 202], [218, 176, 176], [194, 222, 242]]), 'Charming'))

    #R/Lgr Pink Beige
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [236, 217, 202], [218, 196, 148]]), 'Agreeable to the touch'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [184, 190, 189], [245, 223, 181]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [236, 236, 236], [209, 116, 73]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [215, 145, 96], [236, 217, 202]]), 'Smooth'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [245, 223, 181], [180, 180, 180]]), 'Amiable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [245, 223, 181], [197, 188, 213]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [211, 142, 110], [117, 131, 115]]), 'Gentle & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [211, 142, 110], [167, 100, 67]]), 'Mild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[213, 182, 166], [151, 150, 139], [184, 190, 189]]), 'Genteel'))

    #R/L Sandalwood
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [233, 227, 143], [215, 145, 96]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [245, 223, 181], [218, 196, 148]]), 'Mild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [236, 217, 202], [160, 147, 131]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [242, 178, 103], [218, 182, 166]]), 'Domestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [218, 182, 166], [167, 100, 67]]), 'Generous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [206, 185, 179], [158, 128, 110]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [249, 239, 189], [241, 176, 102]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [213, 182, 166], [171, 131, 115]]), 'Gentle & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[211, 142, 110], [206, 185, 179], [151, 150, 139]]), 'Graceful'))

    #R/Gr Rose Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [206, 206, 206], [144, 135, 96]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [211, 142, 110], [218, 196, 148]]), 'Gentle & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [203, 215, 232], [180, 180, 180]]), 'Sedate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [213, 182, 166], [162, 88, 61]]), 'Nostalgic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [218, 196, 148], [148, 133, 105]]), 'Japanese'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [184, 190, 189], [151, 150, 139]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [172, 36, 48], [58, 55, 119]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [115, 71, 79], [206, 185, 179]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[171, 131, 115], [184, 190, 189], [160, 147, 131]]), 'Emotional'))

    #R/DI Old Rose
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [226, 132, 45], [207, 46, 49]]), 'Delicious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [215, 145, 96], [233, 227, 143]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [218, 196, 48], [139, 117, 65]]), 'Nostalgic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [227, 189, 28], [88, 60, 50]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [58, 55, 119], [144, 135, 96]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [16, 76, 84], [109, 116, 73]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [44, 77, 143], [172, 36, 48]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [172, 36, 48], [40, 57, 103]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[162, 88, 61], [158, 128, 110], [54, 88, 48]]), 'Tasteful'))

    #R/Dp Brick Red
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [227, 189, 28], [58, 55, 119]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [158, 128, 110], [40, 57, 103]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [44, 47, 143], [139, 117, 65]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [227, 189, 28], [54, 88, 48]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [227, 189, 28], [10, 10, 10]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [40, 57, 103], [39, 122, 62]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [10, 10, 10], [116, 47, 50]]), 'Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [227, 189, 28], [58, 55, 119]]), 'Dynamic & Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[172, 36, 48], [10, 10, 10], [30, 98, 50]]), 'Untamed'))

    #R/Dk Mahogany
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [162, 88, 61], [215, 145, 96]]), 'Fruitful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [156, 137, 37], [44, 77, 143]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [139, 117, 65], [92, 104, 163]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [162, 88, 61], [58, 55, 119]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [167, 100, 67], [85, 55, 43]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [144, 135, 96], [10, 10, 10]]), 'Substantial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [115, 63, 44], [10, 10, 10]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [79, 46, 43], [125, 126, 126]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[116, 47, 50], [54, 88, 48], [44, 60, 49]]), 'Untamed'))

    #R/Dgr Maroon
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [103, 91, 44], [158, 128, 110]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [231, 47, 39], [255, 200, 8]]), 'Dynamic & Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [103, 91, 44], [126, 126, 126]]), 'Quiet & Sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [231, 47, 39], [227, 189, 28]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [126, 126, 126], [116, 47, 50]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [126, 126, 126], [18, 63, 65]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [156, 137, 37], [172, 36, 48]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [115, 63, 44], [34, 54, 68]]), 'Serious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[79, 46, 43], [86, 86, 86], [25, 62, 63]]), 'Strong & Robust'))

    #YR/V Orange
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [231, 47, 39], [255, 228, 15]]), 'Dazzling'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [167, 100, 67], [242, 178, 103]]), 'Delicious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [255, 236, 79], [43, 151, 89]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [231, 47, 89], [255, 200, 8]]), 'Flamboyant'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [255, 203, 88], [207, 46, 49]]), 'Abudant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [255, 200, 8], [0, 147, 149]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [231, 47, 39], [10, 10, 10]]), 'Forceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [231, 47, 39], [20, 114, 48]]), 'Tropical'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([[238, 113, 25], [236, 236, 236], [3, 86, 155]]), 'Lively'))

    #Yr/S Persimmon
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [242, 178, 103], [169, 87, 49]]), 'Aromatic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [231, 47, 39], [236, 236, 236]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [255, 228, 15], [43, 151, 89]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [207, 46, 49], [162, 88, 61]]), 'Delicious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [207, 46, 49], [227, 189, 28]]), 'Abundant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [207, 46, 49], [115, 71, 79]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [172, 36, 48], [44, 77, 143]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [207, 46, 49], [38, 38, 38]]), 'Dynamic & Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[226, 32, 45], [79, 46, 43], [103, 91, 44]]), 'Untamed'))

    #Yr/B Apricot
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [255, 236, 79], [245, 223, 181]]), 'Delicious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [244, 244, 244], [255, 228, 15]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [249, 239, 189], [88, 171, 45]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [231, 108, 86], [255, 236, 79]]), 'Merry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [255, 236, 79], [146, 198, 131]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [155, 196, 113], [209, 100, 109]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [231, 47, 39], [244, 244, 244]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [231, 47, 39], [3, 86, 155]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[241, 76, 102], [249, 239, 189], [0, 147, 159]]), 'Healthy'))

    #Yr/P Sunset
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [233, 163, 144], [245, 223, 181]]), 'Sweet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [209, 116, 73], [249, 239, 189]]), 'Restful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [219, 220, 93], [245, 223, 181]]), 'Sunny'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [231, 108, 86], [255, 236, 79]]), 'Cheerful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [255, 203, 88], [249, 239, 189]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [146, 198, 131], [249, 239, 189]]), 'Free'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [211, 142, 110], [218, 196, 148]]), 'Domestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [245, 223, 181], [215, 145, 96]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[242, 178, 103], [249, 239, 189], [169, 199, 35]]), 'Friendly'))

    #Yr/Vp Pale Ochre
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [244, 244, 244], [233, 163, 144]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [209, 116, 73], [233, 163, 144]]), 'Domestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [209, 234, 211], [224, 218, 230]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [215, 145, 96], [233, 227, 143]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [242, 178, 103], [219, 220, 93]]), 'Sunny'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [206, 185, 179], [206, 206, 206]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [218, 196, 148], [211, 142, 110]]), 'Mild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [218, 176, 176], [194, 222, 242]]), 'Charming'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[245, 223, 181], [206, 185, 179], [197, 188, 213]]), 'Feminine'))


    rgb_CIS = np.array([[[231, 47, 39], [238, 113, 25], [255, 200, 8], [170, 198, 27], [19, 166, 50], [4, 148, 87], [1, 131, 141], [3, 86, 155], [46, 20, 141], [204, 63, 69]],
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

    #YR/Lgr French Beige
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[5, 2], rgb_CIS[4, 2]]), 'Agreeable to the touch'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[12, 9], rgb_CIS[5, 2]]), 'Amiable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[12, 8], rgb_CIS[4, 6]]), 'Gentle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[12, 8], rgb_CIS[6, 1]]), 'Smooth'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[4, 1], rgb_CIS[5, 1]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[4, 1], rgb_CIS[12, 7]]), 'Dry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[6, 0], rgb_CIS[8, 1]]), 'Generous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[4, 1], rgb_CIS[7, 0]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 1], rgb_CIS[4, 1], rgb_CIS[5, 9]]), 'Mild'))

    #Yr/L Beige
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[3, 1], rgb_CIS[4, 0]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[4, 1], rgb_CIS[5, 1]]), 'Amiable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[4, 2], rgb_CIS[5, 3]]), 'Wholesome'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[3, 0], rgb_CIS[4, 0]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[6, 2], rgb_CIS[5, 2]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[4, 2], rgb_CIS[6, 2]]), 'Generous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[8, 1], rgb_CIS[5, 0]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[7, 0], rgb_CIS[5, 2]]), 'Gentle & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 1], rgb_CIS[4, 1], rgb_CIS[6, 4]]), 'Domestic'))

    #YR/Gr Rose Beige
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[5, 2], rgb_CIS[7, 2]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[12, 7], rgb_CIS[7, 9]]), 'Modest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[5, 7], rgb_CIS[7, 6]]), 'Aqueus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[6, 2], rgb_CIS[8, 2]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[12, 6], rgb_CIS[12, 3]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[12, 5], rgb_CIS[12, 7]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[10, 2], rgb_CIS[11, 0]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[12, 4], rgb_CIS[12, 2]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 1], rgb_CIS[7, 2], rgb_CIS[12, 3]]), 'Sober'))

    #YR/Dl Camel
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[5, 0], rgb_CIS[6, 1]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[12, 5], rgb_CIS[8, 2]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[12, 3], rgb_CIS[8, 2]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[10, 0], rgb_CIS[6, 2]]), 'Aromatic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[10, 1], rgb_CIS[9, 1]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[7, 1], rgb_CIS[11, 1]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[9, 9], rgb_CIS[11, 1]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[10, 2], rgb_CIS[10, 8]]), 'Classic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 1], rgb_CIS[10, 2], rgb_CIS[11, 7]]), 'Traditional'))

    #YR/Dp Brown
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[5, 1], rgb_CIS[0, 1]]), 'Delicious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[10, 1], rgb_CIS[6, 1]]), 'Aromatic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[6, 2], rgb_CIS[9, 2]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[11, 2], rgb_CIS[9, 2]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[11, 0], rgb_CIS[7, 3]]), 'Pratical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[10, 8], rgb_CIS[10, 3]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[1, 0], rgb_CIS[10, 2]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[1, 2], rgb_CIS[12, 0]]), 'Grand'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 1], rgb_CIS[11, 7], rgb_CIS[12, 0]]), 'Sturdy'))

    #YR/Dk Coffee Brown
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[6, 1], rgb_CIS[8, 2]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[8, 1], rgb_CIS[5, 1]]), 'Aromatic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[7, 1], rgb_CIS[12, 4]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[1, 2], rgb_CIS[9, 1]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[8, 1], rgb_CIS[10, 2]]), 'Pratical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[7, 2], rgb_CIS[11, 4]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[10, 0], rgb_CIS[1, 2]]), 'Substantial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[9, 1], rgb_CIS[12, 0]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 1], rgb_CIS[11, 7], rgb_CIS[12, 1]]), 'Heavy & Deep'))

    #YR/Dgr Falcon
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[10, 0], rgb_CIS[8, 1]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[7, 0], rgb_CIS[10, 1]]), 'Pratical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[7, 2], rgb_CIS[10, 3]]), 'Sound'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[8, 1], rgb_CIS[10, 2]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[7, 1], rgb_CIS[11, 2]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[8, 2], rgb_CIS[12, 3]]), 'Quiet & Sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[0, 0], rgb_CIS[12, 0]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[10, 1], rgb_CIS[12, 1]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 1], rgb_CIS[12, 0], rgb_CIS[12, 3]]), 'Strong & Robust'))

    #Y/V Yellow
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 1], rgb_CIS[0, 9]]), 'Vivid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 0], rgb_CIS[0, 4]]), 'Flamboyant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 7], rgb_CIS[0, 9]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 8], rgb_CIS[0, 1]]), 'Dazzling'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 0], rgb_CIS[9, 7]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 0], rgb_CIS[12, 0]]), 'Bold'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 9], rgb_CIS[12, 0]]), 'Dynamic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 2], rgb_CIS[0, 8], rgb_CIS[12, 0]]), 'Intense'))

    #Y/S Gold
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[0, 0], rgb_CIS[10, 1]]), 'Hot'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[9, 0], rgb_CIS[9, 6]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[9, 9], rgb_CIS[1, 8]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[9, 0], rgb_CIS[10, 1]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[8, 0], rgb_CIS[10, 9]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[0, 9], rgb_CIS[0, 8]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[10, 0], rgb_CIS[10, 1]]), 'Substantial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[9, 0], rgb_CIS[10, 8]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 2], rgb_CIS[9, 9], rgb_CIS[9, 8]]), 'Gorgeous'))

    #Y/B Camelia Yellow
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[2, 0], rgb_CIS[3, 0]]), 'Cheerful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[3, 1], rgb_CIS[4, 2]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[12, 9], rgb_CIS[2, 3]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[0, 1], rgb_CIS[2, 5]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[2, 9], rgb_CIS[2, 5]]), 'Pretty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[12, 9], rgb_CIS[2, 5]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[0, 0], rgb_CIS[1, 8]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[0, 0], rgb_CIS[1, 7]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 2], rgb_CIS[2, 4], rgb_CIS[0, 7]]), 'Young'))

    #Y/P Sulphur
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 0], rgb_CIS[3, 1]]), 'Cheerful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 1], rgb_CIS[12, 9]]), 'Healty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[6, 4], rgb_CIS[4, 4]]), 'Citrus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 9], rgb_CIS[3, 4]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[3, 0], rgb_CIS[3, 6]]), 'Childlike'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 4], rgb_CIS[12, 9]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 0], rgb_CIS[2, 4]]), 'Merry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 1], rgb_CIS[3, 5]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 2], rgb_CIS[2, 5], rgb_CIS[2, 7]]), 'Young'))

    #Y/Vp Ivory
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[5, 2], rgb_CIS[5, 1]]), 'Gentle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[3, 1], rgb_CIS[3, 3]]), 'Free'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[6, 2], rgb_CIS[5, 3]]), 'Wholesome'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[3, 1], rgb_CIS[2, 2]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[3, 9], rgb_CIS[4, 6]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[5, 3], rgb_CIS[12, 6]]), 'Plain'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[6, 0], rgb_CIS[3, 0]]), 'Generous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[2, 1], rgb_CIS[2, 4]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 2], rgb_CIS[6, 6], rgb_CIS[5, 4]]), 'Peaceful'))

    #Y/Lgr Light Olive Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[3, 1], rgb_CIS[4, 1]]), 'Restful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[4, 3], rgb_CIS[5, 3]]), 'Plain'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[4, 4], rgb_CIS[5, 4]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[6, 1], rgb_CIS[4, 1]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[6, 2], rgb_CIS[6, 3]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[6, 3], rgb_CIS[7, 3]]), 'Simple & Appealing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[6, 0], rgb_CIS[6, 1]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[6, 1], rgb_CIS[5, 5]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 2], rgb_CIS[12, 6], rgb_CIS[7, 2]]), 'Dry'))

    #Y/L Mustard
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[6, 1], rgb_CIS[5, 2]]), 'Intimate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[6, 1], rgb_CIS[4, 2]]), 'Wholesome'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[4, 2], rgb_CIS[12, 7]]), 'Simple & Appealing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[1, 1], rgb_CIS[5, 2]]), 'Restful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[7, 1], rgb_CIS[5, 2]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[5, 3], rgb_CIS[6, 3]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[9, 1], rgb_CIS[9, 2]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[7, 3], rgb_CIS[8, 2]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 2], rgb_CIS[8, 4], rgb_CIS[10, 8]]), 'Interesting'))

    #Y/Gr Sand Beige
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[5, 1], rgb_CIS[6, 1]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[5, 2], rgb_CIS[12, 6]]), 'Dry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[12, 4], rgb_CIS[12, 6]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[8, 1], rgb_CIS[5, 0]]), 'Nostalgic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[7, 0], rgb_CIS[5, 1]]), 'Japanese'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[12, 2], rgb_CIS[12, 4]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[10, 1], rgb_CIS[10, 3]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[12, 2], rgb_CIS[10, 7]]), 'Sound'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 2], rgb_CIS[10, 6], rgb_CIS[12, 0]]), 'Dapper'))

    #Y/Dl Dusty Olive
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[6, 1], rgb_CIS[10, 1]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[6, 0], rgb_CIS[10, 9]]), 'Nostalgic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[7, 1], rgb_CIS[12, 6]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[10, 0], rgb_CIS[1, 8]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[8, 9], rgb_CIS[10, 5]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[8, 1], rgb_CIS[12, 5]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[10, 0], rgb_CIS[12, 1]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[9, 9], rgb_CIS[10, 8]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 2], rgb_CIS[11, 1], rgb_CIS[12, 3]]), 'Quiet & Sophisticated'))

    #Y/Dp Khaki
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[9, 0], rgb_CIS[10, 1]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[1, 0], rgb_CIS[9, 8]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[6, 2], rgb_CIS[6, 3]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[9, 9], rgb_CIS[11, 9]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[10, 0], rgb_CIS[8, 8]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[9, 1], rgb_CIS[5, 2]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[9, 0], rgb_CIS[10, 8]]), 'Gorgeous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[9, 0], rgb_CIS[10, 2]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 2], rgb_CIS[10, 1], rgb_CIS[12, 2]]), 'Bitter'))

    #Y/Dk Olive
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[7, 0], rgb_CIS[10, 8]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[7, 2], rgb_CIS[12, 2]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[9, 2], rgb_CIS[11, 4]]), 'Bitter'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[11, 1], rgb_CIS[8, 1]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[11, 0], rgb_CIS[12, 4]]), 'Quiet & sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[11, 1], rgb_CIS[10, 8]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[12, 0], rgb_CIS[9, 3]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[10, 9], rgb_CIS[8, 2]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 2], rgb_CIS[12, 4], rgb_CIS[11, 7]]), 'Conservative'))


    #Y/Dgr Olive Brown
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[9, 9], rgb_CIS[8, 1]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[11, 0], rgb_CIS[7, 2]]), 'Quiet & sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[12, 5], rgb_CIS[11, 5]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[9, 2], rgb_CIS[9, 1]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[12, 1], rgb_CIS[9, 2]]), 'Bitter'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[7, 2], rgb_CIS[10, 7]]), 'Dapper'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[10, 0], rgb_CIS[11, 7]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[12, 0], rgb_CIS[10, 1]]), 'Serious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 2], rgb_CIS[12, 4], rgb_CIS[12, 1]]), 'Strong & Robust'))

    #GY/V Yellow Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[0, 1], rgb_CIS[0, 2]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[12, 9], rgb_CIS[0, 2]]), 'Healty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[12, 9], rgb_CIS[4, 4]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[0, 0], rgb_CIS[12, 9]]), 'Festive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[4, 2], rgb_CIS[0, 2]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Clean & Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[0, 1], rgb_CIS[0, 9]]), 'Flamboyant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[2, 9], rgb_CIS[0, 8]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 3], rgb_CIS[0, 9], rgb_CIS[12, 0]]), 'Striking'))

    #GY/S Grass Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[6, 1], rgb_CIS[4, 2]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[6, 2], rgb_CIS[12, 9]]), 'Wholesome'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[4, 2], rgb_CIS[8, 2]]), 'Calm'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[8, 1], rgb_CIS[5, 1]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[3, 0], rgb_CIS[8, 8]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[12, 7], rgb_CIS[7, 4]]), 'Quiet'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[9, 0], rgb_CIS[11, 1]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[8, 9], rgb_CIS[9, 2]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 3], rgb_CIS[12, 7], rgb_CIS[12, 3]]), 'Artistic'))#

    #GY/B Canary
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[12, 9], rgb_CIS[2, 2]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[3, 0], rgb_CIS[2, 2]]), 'Sweet-sour'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[4, 2], rgb_CIS[3, 5]]), 'Free'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[2, 9], rgb_CIS[4, 2]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[3, 1], rgb_CIS[4, 2]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[2, 2], rgb_CIS[0, 0]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[3, 3], rgb_CIS[0, 2]]), 'Citrus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 3], rgb_CIS[4, 3], rgb_CIS[2, 5]]), 'Steady'))

    #Gy/P Lettuce Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[4, 2], rgb_CIS[5, 4]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[4, 2], rgb_CIS[3, 0]]), 'Sweet-sour'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[2, 5], rgb_CIS[4, 6]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[6, 1], rgb_CIS[4, 2]]), 'Free'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[0, 2], rgb_CIS[4, 2]]), 'Citrus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[12, 9], rgb_CIS[2, 5]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[3, 1], rgb_CIS[4, 1]]), 'Healty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[2, 1], rgb_CIS[4, 2]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 3], rgb_CIS[2, 7], rgb_CIS[2, 6]]), 'Youthful'))

    #GY/Vp Pale Chartereuse
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[12, 9], rgb_CIS[4, 0]]), 'Innocent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[12, 9], rgb_CIS[4, 2]]), 'Light'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[5, 2], rgb_CIS[5, 3]]), 'Plain'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[3, 1], rgb_CIS[4, 1]]), 'Sunny'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[4, 0], rgb_CIS[4, 7]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[5, 4], rgb_CIS[4, 6]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[3, 0], rgb_CIS[4, 5]]), 'Sweet-sour'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[4, 8], rgb_CIS[4, 7]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 3], rgb_CIS[2, 3], rgb_CIS[2, 5]]), 'Steady'))

    #GY/Lgr Mist Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[12, 8], rgb_CIS[5, 0]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[12, 6], rgb_CIS[4, 1]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[12, 8], rgb_CIS[5, 7]]), 'Plain'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[4, 1], rgb_CIS[3, 1]]), 'Restful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[6, 1], rgb_CIS[4, 2]]), 'Wholesome'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[6, 2], rgb_CIS[6, 3]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[3, 0], rgb_CIS[4, 2]]), 'Domestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[7, 2], rgb_CIS[6, 3]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 3], rgb_CIS[6, 8], rgb_CIS[7, 1]]), 'Intersting'))

    #GY/Dl Leaf Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[1, 2], rgb_CIS[5, 2]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[12, 6], rgb_CIS[8, 2]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[12, 8], rgb_CIS[8, 8]]), 'Japanese'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[10, 1], rgb_CIS[8, 1]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[3, 0], rgb_CIS[8, 8]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[12, 2], rgb_CIS[12, 6]]), 'Quiet & Sophisticated'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[11, 0], rgb_CIS[0, 0]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[9, 8], rgb_CIS[9, 0]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 3], rgb_CIS[11, 2], rgb_CIS[12, 1]]), 'Diligent'))

    #GY/Dp Olive Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[1, 2], rgb_CIS[1, 0]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[9, 1], rgb_CIS[6, 2]]), 'Pastoral'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[4, 3], rgb_CIS[6, 3]]), 'Natural'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[9, 0], rgb_CIS[1, 2]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[1, 9], rgb_CIS[9, 7]]), 'Complex'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[6, 4], rgb_CIS[9, 6]]), 'Grand'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[9, 8], rgb_CIS[12, 0]]), 'Elaborate'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[11, 1], rgb_CIS[12, 1]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 3], rgb_CIS[12, 6], rgb_CIS[12, 0]]), 'Heavy & Deep'))

    #GY/Dk Ivy Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[10, 0], rgb_CIS[1, 2]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[8, 1], rgb_CIS[5, 3]]), 'Pastoral'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[7, 8], rgb_CIS[12, 3]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[10, 8], rgb_CIS[1, 0]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[8, 0], rgb_CIS[7, 1]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[8, 2], rgb_CIS[12, 2]]), 'Quiet & Sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[11, 1], rgb_CIS[9, 0]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[10, 8], rgb_CIS[9, 1]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 3], rgb_CIS[10, 1], rgb_CIS[7, 2]]), 'Conservative'))

    #GY/Dgr Seaweed
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[10, 1], rgb_CIS[9, 1]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[11, 1], rgb_CIS[7, 1]]), 'Pratical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[8, 6], rgb_CIS[12, 5]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[10, 8], rgb_CIS[8, 2]]), 'Deep'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[10, 2], rgb_CIS[7, 2]]), 'Splendid'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[11, 7], rgb_CIS[12, 5]]), 'Solemn'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[10, 3], rgb_CIS[10, 0]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[8, 2], rgb_CIS[10, 1]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 3], rgb_CIS[11, 1], rgb_CIS[12, 3]]), 'Serious'))

    #G/V Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 0], rgb_CIS[0, 2]]), 'Vigorous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Vivid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[4, 3], rgb_CIS[0, 3]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 0], rgb_CIS[0, 7]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 9], rgb_CIS[0, 1]]), 'Flamboyant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[3, 2], rgb_CIS[0, 8]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 9], rgb_CIS[9, 8]]), 'Provocative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 0], rgb_CIS[12, 0]]), 'Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 4], rgb_CIS[0, 2], rgb_CIS[12, 0]]), 'Dynamic'))

    #G/S Malachite Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[0, 3], rgb_CIS[3, 2]]), 'Youthful'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[4, 2], rgb_CIS[3, 4]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[12, 9], rgb_CIS[3, 5]]), 'Clean & Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Flamboyant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[3, 2], rgb_CIS[1, 7]]), 'Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[5, 3], rgb_CIS[10, 4]]), 'Relaxed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[0, 2], rgb_CIS[12, 0]]), 'Dynamic'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[1, 9], rgb_CIS[12, 0]]), 'Tropical'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 4], rgb_CIS[10, 7], rgb_CIS[12, 7]]), 'Steady'))

    #G/B Emerald
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[2, 1], rgb_CIS[4, 2]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[2, 2], rgb_CIS[4, 2]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[3, 3], rgb_CIS[4, 2]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[2, 0], rgb_CIS[3, 2]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[0, 0], rgb_CIS[4, 9]]), 'Bright'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[2, 9], rgb_CIS[0, 2]]), 'Merry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[0, 0], rgb_CIS[0, 2]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 4], rgb_CIS[0, 9], rgb_CIS[9, 8]]), 'Provocative'))

    #G/P Opaline Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[3, 3], rgb_CIS[4, 2]]), 'Free'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[4, 4], rgb_CIS[12, 9]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[3, 8], rgb_CIS[12, 9]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[4, 1], rgb_CIS[3, 2]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[1, 4], rgb_CIS[4, 2]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[2, 0], rgb_CIS[3, 2]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[1, 7], rgb_CIS[4, 2]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 4], rgb_CIS[12, 8], rgb_CIS[12, 2]]), 'Progressive'))

    #G/Vp Pale Opal
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[12, 9], rgb_CIS[4, 0]]), 'Charming'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[12, 9], rgb_CIS[4, 5]]), 'Neat'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[12, 9], rgb_CIS[3, 5]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[4, 9], rgb_CIS[12, 7]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[3, 2], rgb_CIS[6, 4]]), 'Citrus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Clear'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[3, 9], rgb_CIS[4, 0]]), 'Innocent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[12, 9], rgb_CIS[0, 3]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 4], rgb_CIS[5, 6], rgb_CIS[7, 5]]), 'Quiet'))

    #G/Lgr Ash Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 9], rgb_CIS[5, 0]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 7], rgb_CIS[4, 1]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 8], rgb_CIS[3, 5]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[5, 3], rgb_CIS[4, 2]]), 'Natural'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 6], rgb_CIS[5, 2]]), 'Simple & Appealing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[5, 3], rgb_CIS[4, 3]]), 'Plain'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[3, 1], rgb_CIS[4, 1]]), 'Domestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 6], rgb_CIS[7, 5]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 4], rgb_CIS[12, 9], rgb_CIS[3, 7]]), 'Crystalline'))

    #G/L Spray Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[12, 7], rgb_CIS[4, 1]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[3, 3], rgb_CIS[4, 3]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[12, 9], rgb_CIS[4, 5]]), 'Neat'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[3, 2], rgb_CIS[4, 4]]), 'Citrus'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[7, 5], rgb_CIS[4, 4]]), 'Tranquil'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[6, 1], rgb_CIS[4, 2]]), 'Healty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[4, 8], rgb_CIS[6, 7]]), 'Western'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 4], rgb_CIS[9, 3], rgb_CIS[9, 6]]), 'Grand'))

    #G/Gr Mst Green II
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 5], rgb_CIS[5, 2]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 7], rgb_CIS[5, 6]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 8], rgb_CIS[7, 6]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[7, 8], rgb_CIS[5, 7]]), 'Fashionable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 7], rgb_CIS[7, 8]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 9], rgb_CIS[8, 7]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[8, 9], rgb_CIS[5, 8]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[12, 8], rgb_CIS[8, 8]]), 'Stylish'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 4], rgb_CIS[4, 5], rgb_CIS[11, 9]]), 'Grand'))

    #G/Dl Jade Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[1, 1], rgb_CIS[1, 0]]), 'Abundant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[7, 3], rgb_CIS[6, 2]]), 'Natural'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[8, 7], rgb_CIS[12, 9]]), 'Intellectual'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[5, 9], rgb_CIS[11, 9]]), 'Adult'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[10, 2], rgb_CIS[12, 8]]), 'Tranquil'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[5, 5], rgb_CIS[7, 7]]), 'Quiet'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[1, 9], rgb_CIS[10, 0]]), 'Complex'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[6, 2], rgb_CIS[10, 8]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 4], rgb_CIS[7, 3], rgb_CIS[11, 5]]), 'Deep'))#

    #G/Dp Viridian
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[0, 9], rgb_CIS[0, 1]]), 'Tropical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[5, 2], rgb_CIS[7, 4]]), 'Natural'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[12, 7], rgb_CIS[7, 4]]), 'Tranquil'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[10, 2], rgb_CIS[1, 0]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[9, 8], rgb_CIS[12, 6]]), 'Decorative'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[4, 5], rgb_CIS[9, 7]]), 'Progressive'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[9, 9], rgb_CIS[1, 2]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[12, 0], rgb_CIS[9, 1]]), 'Pratical'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 4], rgb_CIS[12, 1], rgb_CIS[7, 2]]), 'Sound'))#

    #G/Dk Bottle Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[0, 0], rgb_CIS[0, 1]]), 'Abundant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[5, 8], rgb_CIS[8, 9]]), 'Adult'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[12, 6], rgb_CIS[7, 5]]), 'Quiet'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[9, 0], rgb_CIS[1, 2]]), 'Rich'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[10, 1], rgb_CIS[8, 1]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[11, 1], rgb_CIS[12, 6]]), 'Dignified'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[12, 0], rgb_CIS[9, 0]]), 'Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[10, 8], rgb_CIS[9, 1]]), 'Classic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 4], rgb_CIS[11, 0], rgb_CIS[8, 2]]), 'Pratical'))

    #G/Dgr Jungle Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[12, 4], rgb_CIS[11, 1]]), 'Quiet & Sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[9, 2], rgb_CIS[10, 2]]), 'Bitter'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[11, 8], rgb_CIS[7, 7]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[11, 0], rgb_CIS[7, 2]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[10, 1], rgb_CIS[7, 2]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[12, 4], rgb_CIS[12, 0]]), 'Strong & Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[12, 0], rgb_CIS[10, 1]]), 'Heavy'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[11, 0], rgb_CIS[8, 2]]), 'Sound'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 4], rgb_CIS[10, 1], rgb_CIS[11, 7]]), 'Heavy & Deep'))

    #BG/V Peacock Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[0, 1], rgb_CIS[3, 2]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[12, 7], rgb_CIS[10, 7]]), 'Progressive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[0, 1], rgb_CIS[0, 9]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[0, 2], rgb_CIS[10, 7]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[12, 7], rgb_CIS[12, 1]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[0, 0], rgb_CIS[12, 0]]), 'Vigorous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[2, 2], rgb_CIS[12, 0]]), 'Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 5], rgb_CIS[12, 9], rgb_CIS[12, 1]]), 'Sharp'))

    #BG/S Jewel Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[12, 9], rgb_CIS[0, 0]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[3, 2], rgb_CIS[10, 7]]), 'Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[12, 9], rgb_CIS[9, 7]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[0, 9], rgb_CIS[2, 1]]), 'Tropical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[5, 6], rgb_CIS[10, 5]]), 'Grand'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[12, 7], rgb_CIS[8, 7]]), 'Smart'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[0, 8], rgb_CIS[0, 9]]), 'Provocative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[9, 8], rgb_CIS[1, 7]]), 'Mysterious'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 5], rgb_CIS[12, 9], rgb_CIS[10, 7]]), 'Sharp'))

    #BG/B Turquoise
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[3, 0], rgb_CIS[3, 2]]), 'Pretty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[2, 3], rgb_CIS[3, 2]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[1, 1], rgb_CIS[2, 2]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[0, 6], rgb_CIS[3, 2]]), 'Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Clean & Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Tropical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[2, 9], rgb_CIS[0, 8]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 5], rgb_CIS[4, 4], rgb_CIS[0, 7]]), 'Progressive'))

    #BG/P Light Aqua Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[3, 2], rgb_CIS[4, 2]]), 'Healty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[12, 9], rgb_CIS[4, 3]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[12, 9], rgb_CIS[4, 5]]), 'Neat'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[1, 4], rgb_CIS[12, 9]]), 'Clean & Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[4, 6], rgb_CIS[2, 7]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Clean'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[4, 0], rgb_CIS[8, 8]]), 'Western'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[12, 6], rgb_CIS[9, 7]]), 'Cultivated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 5], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Crystalline'))

    #BG/Vp Horizon Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[12, 8], rgb_CIS[4, 0]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[12, 9], rgb_CIS[4, 7]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[12, 9], rgb_CIS[3, 5]]), 'Neat'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[4, 1], rgb_CIS[4, 3]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[4, 8], rgb_CIS[5, 7]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[12, 9], rgb_CIS[3, 7]]), 'Clean'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[5, 1], rgb_CIS[4, 2]]), 'Gentle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[3, 6], rgb_CIS[8, 8]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 5], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Crystalline'))

    #BG/Lgr Eggshell Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[3, 9], rgb_CIS[4, 2]]), 'Pretty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[5, 1], rgb_CIS[4, 2]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Neat'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[3, 9], rgb_CIS[4, 9]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[6, 1], rgb_CIS[5, 2]]), 'Friendly'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[12, 9], rgb_CIS[6, 7]]), 'Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[2, 1], rgb_CIS[3, 2]]), 'Lighthearted'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[8, 8], rgb_CIS[12, 5]]), 'Stylish'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 5], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Pure'))

    #BG/L Venice Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[6, 0], rgb_CIS[3, 2]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[12, 9], rgb_CIS[5, 3]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[12, 9], rgb_CIS[4, 5]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[2, 9], rgb_CIS[2, 2]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[4, 5], rgb_CIS[8, 6]]), 'Dewy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[4, 6], rgb_CIS[8, 7]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[1, 9], rgb_CIS[0, 8]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[4, 1], rgb_CIS[8, 7]]), 'Cultivated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 5], rgb_CIS[12, 9], rgb_CIS[8, 7]]), 'Western'))

    #BG/Gr Blue Spruce
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[4, 4], rgb_CIS[5, 6]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[4, 6], rgb_CIS[5, 6]]), 'Dewy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[12, 8], rgb_CIS[12, 4]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[7, 2], rgb_CIS[12, 6]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[8, 8], rgb_CIS[12, 7]]), 'Noble'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[12, 6], rgb_CIS[12, 3]]), 'Aqueous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[4, 2], rgb_CIS[10, 6]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[12, 0], rgb_CIS[8, 8]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 5], rgb_CIS[12, 7], rgb_CIS[10, 6]]), 'Intellectual'))

    #BG/Dl Cambridge Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[8, 9], rgb_CIS[1, 2]]), 'Ethnic'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[12, 5], rgb_CIS[11, 7]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[12, 9], rgb_CIS[9, 7]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[10, 8], rgb_CIS[1, 9]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[0, 8], rgb_CIS[1, 9]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[12, 9], rgb_CIS[11, 8]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[9, 0], rgb_CIS[10, 8]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[10, 3], rgb_CIS[8, 8]]), 'Complex'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 5], rgb_CIS[5, 6], rgb_CIS[10, 6]]), 'Magnificent'))#

    #BG/Dp Prussian Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[12, 8], rgb_CIS[9, 0]]), 'Casual'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[5, 8], rgb_CIS[8, 7]]), 'Smart'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[5, 6], rgb_CIS[4, 6]]), 'Tranquil'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[6, 1], rgb_CIS[6, 2]]), 'Friendly'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[7, 3], rgb_CIS[10, 8]]), 'Elaborate'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[12, 9], rgb_CIS[10, 7]]), 'Intellectual'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[0, 0], rgb_CIS[11, 0]]), 'Wild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[10, 0], rgb_CIS[1, 2]]), 'Ethnic'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 5], rgb_CIS[11, 3], rgb_CIS[8, 3]]), 'Deep'))#

    #BG/Dk Teal Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[9, 2], rgb_CIS[9, 0]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[10, 1], rgb_CIS[7, 2]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[12, 4], rgb_CIS[7, 6]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[8, 9], rgb_CIS[8, 2]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[12, 0], rgb_CIS[7, 2]]), 'Intrepid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[12, 0], rgb_CIS[12, 6]]), 'Metallic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[10, 9], rgb_CIS[1, 2]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[12, 0], rgb_CIS[1, 2]]), 'Untamed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 5], rgb_CIS[10, 8], rgb_CIS[12, 0]]), 'Majestic'))

    #BG/Dkr Dusky Green
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[5, 1], rgb_CIS[7, 2]]), 'Calm'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[7, 5], rgb_CIS[8, 5]]), 'Magnificent'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[4, 6], rgb_CIS[8, 7]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[11, 0], rgb_CIS[7, 1]]), 'Classic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[11, 8], rgb_CIS[7, 2]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[11, 2], rgb_CIS[12, 4]]), 'Placid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[9, 0], rgb_CIS[12, 0]]), 'Untamed'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[10, 2], rgb_CIS[11, 6]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 5], rgb_CIS[12, 0], rgb_CIS[10, 1]]), 'Serious'))

    #B/V Cerulean Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[0, 0], rgb_CIS[0, 2]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[0, 0], rgb_CIS[12, 9]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[2, 5], rgb_CIS[3, 2]]), 'Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Vivid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[0, 8], rgb_CIS[0, 9]]), 'Provocative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Speedy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[0, 0], rgb_CIS[12, 0]]), 'Active'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[12, 0], rgb_CIS[0, 9]]), 'Bold'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 6], rgb_CIS[12, 0], rgb_CIS[12, 9]]), 'Sharp'))

    #B/S Light Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 9], rgb_CIS[3, 2]]), 'Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[4, 6], rgb_CIS[3, 5]]), 'Fresh & Young'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 9], rgb_CIS[3, 7]]), 'Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[0, 0], rgb_CIS[2, 2]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 7], rgb_CIS[1, 8]]), 'Noble'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 9], rgb_CIS[9, 7]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 9], rgb_CIS[0, 9]]), 'Sporty'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[10, 7], rgb_CIS[3, 6]]), 'Masculine'))#
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 6], rgb_CIS[12, 7], rgb_CIS[12, 0]]), 'Modern'))

    #B/B Sky Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[0, 1], rgb_CIS[4, 2]]), 'Open'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 9], rgb_CIS[2, 3]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 9], rgb_CIS[3, 4]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[2, 9], rgb_CIS[2, 2]]), 'Enjoyable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[2, 5], rgb_CIS[4, 5]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 0], rgb_CIS[0, 0]]), 'Striking'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 0], rgb_CIS[12, 9]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 6], rgb_CIS[12, 9], rgb_CIS[0, 7]]), 'Speedy'))

    #B/P Aqua Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[4, 4]]), 'Clear'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[3, 1], rgb_CIS[4, 2]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[4, 6]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[3, 0], rgb_CIS[3, 2]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[7, 6]]), 'Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[3, 7]]), 'Clean'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[2, 0], rgb_CIS[4, 2]]), 'Childlike'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[2, 5]]), 'Clean & Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 6], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Smart'))

    #B/Vp Pale Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[3, 9], rgb_CIS[4, 9]]), 'Charming'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[4, 9], rgb_CIS[12, 9]]), 'Romantic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[3, 5], rgb_CIS[12, 9]]), 'Clear'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[3, 9], rgb_CIS[4, 2]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[12, 9], rgb_CIS[2,7]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[3, 8], rgb_CIS[4, 1]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[2, 5], rgb_CIS[12, 9]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 6], rgb_CIS[12, 9], rgb_CIS[2, 6]]), 'Refreshing'))

    ##B/Lgr Powder Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[4, 1], rgb_CIS[5, 8]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[12, 9], rgb_CIS[6, 2]]), 'Salty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[6, 6], rgb_CIS[4, 7]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[7, 8], rgb_CIS[12, 9]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[12, 7], rgb_CIS[7, 4]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[4, 6], rgb_CIS[12, 5]]), 'Dewy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[9, 8], rgb_CIS[11, 7]]), 'Eminent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[7, 8], rgb_CIS[12, 3]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 6], rgb_CIS[7, 6], rgb_CIS[12, 2]]), 'Distinguished'))

    #B/L Aquamarine
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[5, 4], rgb_CIS[4, 2]]), 'Peaceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[5, 3], rgb_CIS[6, 8]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[5, 6], rgb_CIS[4, 7]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[1, 8], rgb_CIS[5, 8]]), 'Noble & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[6, 7], rgb_CIS[8, 7]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[12, 8], rgb_CIS[8, 7]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[12, 0], rgb_CIS[8, 8]]), 'Stylish'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[4, 2], rgb_CIS[11, 7]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 6], rgb_CIS[5, 7], rgb_CIS[11, 6]]), 'Precise'))

    #B/Gr Blue Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 7], rgb_CIS[12, 3]]), 'Precise'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 7], rgb_CIS[8, 7]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 1], rgb_CIS[4, 6]]), 'Composed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[8, 7], rgb_CIS[12, 2]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 1], rgb_CIS[9, 7]]), 'Sublime'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 2], rgb_CIS[5, 6]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[10, 8], rgb_CIS[11, 4]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 4], rgb_CIS[11, 7]]), 'Solemn'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 6], rgb_CIS[12, 1], rgb_CIS[12, 0]]), 'Authoritative'))

    #B/Dl Shadow Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[3, 7], rgb_CIS[7, 7]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 4], rgb_CIS[12, 9]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 8], rgb_CIS[9, 7]]), 'Rational'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 3], rgb_CIS[7, 5]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 7], rgb_CIS[12, 0]]), 'Composed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 0], rgb_CIS[12, 1]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[12, 1], rgb_CIS[7, 2]]), 'Pratical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[11, 3], rgb_CIS[12, 5]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 6], rgb_CIS[5, 7], rgb_CIS[10, 7]]), 'Precise'))

    #B/Dp Peacock Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[9, 0], rgb_CIS[1, 2]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[12, 8], rgb_CIS[7, 7]]), 'Composed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[12, 9], rgb_CIS[2, 7]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[0, 9], rgb_CIS[9, 8]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[12, 7], rgb_CIS[7, 6]]), 'Precise'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[10, 7], rgb_CIS[0, 7]]), 'Masuline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[9, 2], rgb_CIS[11, 7]]), 'Intrepid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[12, 8], rgb_CIS[9, 8]]), 'Precious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 6], rgb_CIS[12, 8], rgb_CIS[11, 8]]), 'Metallic'))

    #B/Dk Teal
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 9], rgb_CIS[0, 2]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[7, 7], rgb_CIS[5, 7]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 7], rgb_CIS[1, 7]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 7], rgb_CIS[7, 5]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[8, 7], rgb_CIS[5, 7]]), 'Rational'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 4], rgb_CIS[12, 0]]), 'Earnest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[10, 1], rgb_CIS[8, 2]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 4], rgb_CIS[11, 0]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 6], rgb_CIS[12, 0], rgb_CIS[7, 2]]), 'Dapper'))

    #B/Dgr Prussian Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[8, 2], rgb_CIS[8, 6]]), 'Dapper'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[7, 2], rgb_CIS[8, 8]]), 'Sublime'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[4, 7], rgb_CIS[2, 7]]), 'Progressive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[11, 2], rgb_CIS[9, 0]]), 'Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[11, 2], rgb_CIS[7, 2]]), 'Dapper'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[7, 8], rgb_CIS[6, 6]]), 'Precise'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[12, 1], rgb_CIS[10, 9]]), 'Strong & Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[11, 5], rgb_CIS[10, 2]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 6], rgb_CIS[12, 0], rgb_CIS[7, 8]]), 'Solemn'))

    #PB/V Ultramarine
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[0, 0], rgb_CIS[2, 1]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[12, 9], rgb_CIS[2, 5]]), 'Clean & fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[12, 9], rgb_CIS[2, 6]]), 'Nimble'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[0, 0], rgb_CIS[0, 2]]), 'Vigorous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[2, 2], rgb_CIS[12, 9]]), 'Fleet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[2, 5], rgb_CIS[4, 4]]), 'Progressive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Striking'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[12, 0], rgb_CIS[0, 2]]), 'Intense'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 7], rgb_CIS[12, 0], rgb_CIS[12, 9]]), 'Sharp'))

    #PB/S Sapphire
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[12, 9], rgb_CIS[3, 2]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[12, 9], rgb_CIS[4, 5]]), 'Neat'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[12, 9], rgb_CIS[4, 6]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[3, 5], rgb_CIS[3, 2]]), 'Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[6, 7], rgb_CIS[4, 7]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[12, 9], rgb_CIS[2, 6]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[3, 4], rgb_CIS[4, 2]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[5, 7], rgb_CIS[12, 0]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 7], rgb_CIS[12, 9], rgb_CIS[12, 0]]), 'Intellectual'))

    #PB/B salvia Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[2, 2]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[4, 4]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Clear'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[2, 3]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[4, 6], rgb_CIS[3, 5]]), 'Fresh & Young'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[4, 6]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[3, 5]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[12, 9], rgb_CIS[1, 6]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 7], rgb_CIS[11, 6], rgb_CIS[4, 7]]), 'Progressive'))

    #PB/P Sky Mist
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[2, 0], rgb_CIS[2, 2]]), 'Casual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[12, 8], rgb_CIS[4, 6]]), 'Refined'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[12, 9], rgb_CIS[3, 6]]), 'Clean'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[12, 9], rgb_CIS[3, 8]]), 'Pure & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[12, 8], rgb_CIS[7, 7]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[4, 7], rgb_CIS[3, 5]]), 'Refreshing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[10, 7], rgb_CIS[8, 8]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[7, 7], rgb_CIS[8, 6]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 7], rgb_CIS[12, 9], rgb_CIS[1, 6]]), 'Simple'))

    #PB/Vp Pale Mist
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[12, 8], rgb_CIS[5, 9]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[12, 9], rgb_CIS[4, 2]]), 'Light'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[12, 9], rgb_CIS[4, 6]]), 'Pure'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[4, 8], rgb_CIS[4, 4]]), 'Romantic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[7, 7], rgb_CIS[5, 8]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[5, 6], rgb_CIS[3, 7]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[3, 9], rgb_CIS[4, 1]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[12, 6], rgb_CIS[7, 8]]), 'Sedate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 7], rgb_CIS[10, 7], rgb_CIS[7, 6]]), 'Composed'))

    #PB/Lgr Moonstone Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 8], rgb_CIS[5, 8]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 5], rgb_CIS[4, 4]]), 'Fashionable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 8], rgb_CIS[5, 6]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[4, 8], rgb_CIS[5, 9]]), 'Subtle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 7], rgb_CIS[7, 7]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 8], rgb_CIS[12, 5]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 1], rgb_CIS[7, 9]]), 'Noble & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[12, 3], rgb_CIS[8, 7]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 7], rgb_CIS[6, 6], rgb_CIS[11, 6]]), 'Precise'))

    #PB/L Pale Blue II
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 7], rgb_CIS[12, 9]]), 'Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[4, 6], rgb_CIS[3, 8]]), 'Pure & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[4, 6], rgb_CIS[12, 9]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 7], rgb_CIS[7, 8]]), 'Refined'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 3], rgb_CIS[5, 8]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 2], rgb_CIS[4, 7]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[11, 6], rgb_CIS[1, 8]]), 'Aristocratic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 3], rgb_CIS[7, 7]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 7], rgb_CIS[12, 1], rgb_CIS[1, 7]]), 'Intellectual'))

    #PB/Gr Slate Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 6], rgb_CIS[5, 9]]), 'Refined'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[5, 8], rgb_CIS[6, 8]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 7], rgb_CIS[3, 7]]), 'Aqueous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 7], rgb_CIS[7, 1]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[4, 7], rgb_CIS[5, 8]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 7], rgb_CIS[7, 8]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 1], rgb_CIS[10, 8]]), 'Majestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 3], rgb_CIS[5, 6]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 7], rgb_CIS[12, 7], rgb_CIS[10, 7]]), 'Precise'))

    #PB/Dl Shadow Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 6], rgb_CIS[4, 2]]), 'Progressive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[4, 4], rgb_CIS[1, 4]]), 'Steady'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 9], rgb_CIS[3, 5]]), 'Western'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 6], rgb_CIS[7, 8]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 7], rgb_CIS[7, 6]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 2], rgb_CIS[4, 8]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[6, 7], rgb_CIS[6, 6]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 6], rgb_CIS[10, 7]]), 'Cultivated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 7], rgb_CIS[12, 1], rgb_CIS[12, 9]]), 'Masculine'))

    #PB/Dp Mineral Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[0, 1], rgb_CIS[0, 0]]), 'Lively'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 9], rgb_CIS[7, 5]]), 'Smart'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 9], rgb_CIS[2, 6]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[0, 2], rgb_CIS[0, 0]]), 'Vigorous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 5], rgb_CIS[11, 7]]), 'Composed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 8], rgb_CIS[8, 6]]), 'Rational'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 1], rgb_CIS[7, 6]]), 'Sublime'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 8], rgb_CIS[11, 8]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 7], rgb_CIS[12, 1], rgb_CIS[12, 6]]), 'Earnest'))

    #PB/Dk dark Mineral Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[8, 8], rgb_CIS[3, 7]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 5], rgb_CIS[4, 7]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 8], rgb_CIS[0, 2]]), 'Agile'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[5, 6], rgb_CIS[8, 8]]), 'Eminent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 7], rgb_CIS[7, 7]]), 'Precise'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 9], rgb_CIS[2, 5]]), 'Sharp'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 5], rgb_CIS[10, 1]]), 'Earnest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[9, 6], rgb_CIS[4, 7]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 7], rgb_CIS[12, 0], rgb_CIS[12, 5]]), 'Formal'))

    #PB/Dgr Midnight Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[11, 2], rgb_CIS[7, 2]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[10, 1], rgb_CIS[7, 3]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 8], rgb_CIS[12, 4]]), 'Metallic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 0], rgb_CIS[9, 1]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 4], rgb_CIS[7, 6]]), 'Solemn'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 8], rgb_CIS[9, 7]]), 'Proper'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 4], rgb_CIS[11, 0]]), 'Strong & robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 1], rgb_CIS[10, 1]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 7], rgb_CIS[12, 1], rgb_CIS[7, 6]]), 'Authoritative'))

    #P/V Purple
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[0, 2], rgb_CIS[0, 1]]), 'Dazzling'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[2, 0], rgb_CIS[0, 9]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[2, 9], rgb_CIS[0, 3]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[0, 9], rgb_CIS[0, 2]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[0, 9], rgb_CIS[3, 0]]), 'Fascinating'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[1, 9], rgb_CIS[6, 5]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[1, 2], rgb_CIS[9, 0]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[3, 8], rgb_CIS[2, 9]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 8], rgb_CIS[0, 9], rgb_CIS[2, 4]]), 'Provocative'))

    #P/S Violet
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[3, 9], rgb_CIS[1, 9]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[5, 9], rgb_CIS[6, 9]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[3, 7], rgb_CIS[7, 7]]), 'Noble & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[2, 0], rgb_CIS[1, 9]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[3, 4], rgb_CIS[1, 9]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[5, 7], rgb_CIS[10, 7]]), 'Eminent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[8, 2], rgb_CIS[10, 0]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[7, 8], rgb_CIS[12, 1]]), 'Aristocratic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 8], rgb_CIS[12, 0], rgb_CIS[10, 7]]), 'Precious'))

    #P/B Lavander
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[2, 2], rgb_CIS[2, 0]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[12, 7], rgb_CIS[3, 7]]), 'Refined'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[3, 8], rgb_CIS[4, 7]]), 'Pure & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[3, 3], rgb_CIS[2, 0]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[3, 7], rgb_CIS[1, 8]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[3, 8], rgb_CIS[7, 7]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[0, 9], rgb_CIS[0, 8]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[1, 9], rgb_CIS[11, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 8], rgb_CIS[12, 9], rgb_CIS[12, 0]]), 'Modern'))

    #P/P Lilac
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[3, 9], rgb_CIS[4, 0]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[3, 0], rgb_CIS[4, 9]]), 'Sweet & Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[12, 9], rgb_CIS[5, 7]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[6, 9], rgb_CIS[4, 8]]), 'Feminine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[4, 9], rgb_CIS[7, 8]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[4, 6], rgb_CIS[6, 7]]), 'Pure & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[2, 9], rgb_CIS[8, 8]]), 'Fascinating'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[2, 8], rgb_CIS[1, 8]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 8], rgb_CIS[7, 7], rgb_CIS[5, 8]]), 'Sleek'))

    #P/Vp Pale Lilac
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[5, 0], rgb_CIS[4, 5]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[4, 2], rgb_CIS[4, 6]]), 'Romantic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[12, 7], rgb_CIS[4, 6]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[3, 1], rgb_CIS[3, 9]]), 'Sweet & dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[5, 9], rgb_CIS[5, 8]]), 'Cultured'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[4, 5], rgb_CIS[5, 7]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[6, 9], rgb_CIS[3, 8]]), 'Feminine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[12, 6], rgb_CIS[5, 3]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 8], rgb_CIS[5, 8], rgb_CIS[5, 7]]), 'Subtle'))

    #P/Lgr Stalight Blue
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[5, 9], rgb_CIS[5, 7]]), 'Sedate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[5, 0], rgb_CIS[4, 9]]), 'Gentle & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[4, 8], rgb_CIS[5, 7]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[4, 9], rgb_CIS[7, 9]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[5, 9], rgb_CIS[7, 8]]), 'Cultured'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[4, 6], rgb_CIS[7, 7]]), 'Noble'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[6, 8], rgb_CIS[7, 7]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[12, 7], rgb_CIS[7, 8]]), 'Subtle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 8], rgb_CIS[12, 5], rgb_CIS[12, 3]]), 'Chic'))

    #P/L Lilac II
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[3, 9], rgb_CIS[4, 1]]), 'Feminine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[4, 9], rgb_CIS[4, 5]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[12, 8], rgb_CIS[6, 6]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[6, 9], rgb_CIS[4, 8]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[3, 9], rgb_CIS[7, 8]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[7, 8], rgb_CIS[4, 5]]), 'Fashionable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[1, 9], rgb_CIS[8, 8]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[7, 7], rgb_CIS[5, 9]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 8], rgb_CIS[5, 3], rgb_CIS[7, 5]]), 'Interesting'))

    #P/Gr Pigeon
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[6, 0], rgb_CIS[5, 9]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[12, 7], rgb_CIS[6, 8]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[5, 8], rgb_CIS[6, 7]]), 'Refined'))# 81
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[7, 3], rgb_CIS[5, 2]]), 'Japanese'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[5, 9], rgb_CIS[7, 6]]), 'Fashionable'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[12, 8], rgb_CIS[12, 5]]), 'Sedate'))# 67
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[8, 2], rgb_CIS[5, 7]]), 'Subtle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[8, 9], rgb_CIS[12, 6]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 8], rgb_CIS[12, 6], rgb_CIS[12, 1]]), 'Subtle & Mysterious'))# 168

    #P/Dl Dusty lilac
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[6, 2], rgb_CIS[12, 6]]), 'Stylish'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[7, 5], rgb_CIS[12, 7]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[5, 8], rgb_CIS[6, 7]]), 'Noble & Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[1, 9], rgb_CIS[6, 8]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[7, 7], rgb_CIS[10, 7]]), 'Aristocratic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[6, 7], rgb_CIS[12, 0]]), 'Eminent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[3, 0], rgb_CIS[8, 3]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[12, 0], rgb_CIS[12, 4]]), 'Precious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 8], rgb_CIS[12, 1], rgb_CIS[12, 7]]), 'Sublime'))

    #P/Dp Pansy
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[0, 2], rgb_CIS[0, 9]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[12, 7], rgb_CIS[1, 9]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[5, 6], rgb_CIS[11, 7]]), 'Eminent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[9, 0], rgb_CIS[1, 2]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[9, 0], rgb_CIS[7, 0]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[10, 0], rgb_CIS[9, 2]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[9, 9], rgb_CIS[1, 2]]), 'Gorgeous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[1, 9], rgb_CIS[8, 9]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 8], rgb_CIS[12, 0], rgb_CIS[7, 1]]), 'Precious'))

    #P/Dk Prune
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[9, 9], rgb_CIS[8, 2]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[10, 2], rgb_CIS[7, 0]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[12, 6], rgb_CIS[7, 3]]), 'Modest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[9, 0], rgb_CIS[8, 5]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[10, 2], rgb_CIS[8, 1]]), 'Classic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[12, 7], rgb_CIS[7, 6]]), 'Sublime'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[1, 9], rgb_CIS[8, 5]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[9, 1], rgb_CIS[10, 3]]), 'Traditional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 8], rgb_CIS[12, 0], rgb_CIS[7, 7]]), 'Majestic'))

    #P/Dgr Dusky Violet
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[1, 9], rgb_CIS[2, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 7], rgb_CIS[6, 7]]), 'Rational'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 9], rgb_CIS[8, 5]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 6], rgb_CIS[11, 1]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[7, 6], rgb_CIS[11, 4]]), 'Dignified'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 9], rgb_CIS[8, 7]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[7, 2], rgb_CIS[11, 5]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 1], rgb_CIS[10, 2]]), 'Serious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 8], rgb_CIS[12, 5], rgb_CIS[12, 2]]), 'Solemn'))

    #RP/V Magenta
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[12, 9], rgb_CIS[0, 2]]), 'Dazzling'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[0, 4], rgb_CIS[0, 2]]), 'Vivid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[0, 6], rgb_CIS[0, 2]]), 'Flamboyant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[2, 1], rgb_CIS[1, 5]]), 'Tropical'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[0, 8], rgb_CIS[0, 2]]), 'Showy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[1, 2], rgb_CIS[9, 8]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[12, 0], rgb_CIS[0, 2]]), 'Dynamic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[12, 0], rgb_CIS[0, 3]]), 'Fiery'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[0, 9], rgb_CIS[0, 8], rgb_CIS[1, 5]]), 'Provocative'))

    #RP/S Spinner Red
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[0, 0], rgb_CIS[0, 1]]), 'Abundant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[3, 9], rgb_CIS[8, 8]]), 'Brilliant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[12, 8], rgb_CIS[6, 6]]), 'Western'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[2, 9], rgb_CIS[0, 8]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[9, 8], rgb_CIS[0, 9]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[3, 4], rgb_CIS[1, 8]]), 'Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[0, 2], rgb_CIS[0, 8]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[12, 0], rgb_CIS[8, 8]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[1, 9], rgb_CIS[8, 8], rgb_CIS[9, 6]]), 'Complex'))

    #RP/B Rose Pink
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[3, 2], rgb_CIS[2, 5]]), 'Cute'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[2, 1], rgb_CIS[3, 4]]), 'Amusing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[3, 2], rgb_CIS[3, 6]]), 'Childlike'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[4, 8], rgb_CIS[0, 0]]), 'Happy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[2, 1], rgb_CIS[0, 3]]), 'Merry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[4, 2], rgb_CIS[2, 7]]), 'Pretty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[3, 8], rgb_CIS[8, 8]]), 'Fascinating'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[2, 5], rgb_CIS[0, 8]]), 'Colorful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[2, 9], rgb_CIS[3, 3], rgb_CIS[1, 7]]), 'Enjoyable'))

    #RP/P Mauve Pink
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[12, 9], rgb_CIS[3, 2]]), 'Pretty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 0], rgb_CIS[4, 8]]), 'Soft'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 4], rgb_CIS[4, 0]]), 'Innocent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[3, 0], rgb_CIS[4, 1]]), 'Sweet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 1], rgb_CIS[4, 5]]), 'Charming'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 2], rgb_CIS[4, 6]]), 'Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 0], rgb_CIS[3, 8]]), 'Sweet & Dreamy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[4, 9], rgb_CIS[5, 5]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[3, 9], rgb_CIS[6, 8], rgb_CIS[4, 5]]), 'Feminine'))

    #RP/Vp Cherry Rose
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[5, 0], rgb_CIS[4, 1]]), 'Agreeable to the touch'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[12, 9], rgb_CIS[4, 4]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[12, 9], rgb_CIS[4, 7]]), 'Romantic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[3, 1], rgb_CIS[4, 2]]), 'Sunny'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[3, 9], rgb_CIS[4, 2]]), 'Innocent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[3, 9], rgb_CIS[4, 6]]), 'Charming'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[5, 1], rgb_CIS[6, 0]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[5, 0], rgb_CIS[5, 6]]), 'Gentle & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[4, 9], rgb_CIS[4, 5], rgb_CIS[6, 8]]), 'Emotional'))

    #RP/Lgr Rose Mist
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[5, 1], rgb_CIS[4, 1]]), 'Mild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[3, 8], rgb_CIS[4, 1]]), 'Feminine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[12, 8], rgb_CIS[6, 8]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[4, 8], rgb_CIS[5, 8]]), 'Cultured'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[4, 8], rgb_CIS[12, 6]]), 'Delicate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[4, 1], rgb_CIS[5, 7]]), 'Supple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[6, 0], rgb_CIS[7, 1]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[4, 8], rgb_CIS[7, 8]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[5, 9], rgb_CIS[12, 6], rgb_CIS[7, 7]]), 'Refined'))

    #RP/L Orchid
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[7, 0], rgb_CIS[5, 0]]), 'Pleasant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[4, 8], rgb_CIS[6, 8]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[3, 8], rgb_CIS[4, 8]]), 'Feminine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[7, 0], rgb_CIS[10, 3]]), 'Nostalgic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[7, 8], rgb_CIS[8, 9]]), 'Sedate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[7, 9], rgb_CIS[8, 8]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[0, 9], rgb_CIS[1, 8]]), 'Glossy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[8, 9], rgb_CIS[8, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[6, 9], rgb_CIS[4, 7], rgb_CIS[10, 7]]), 'Western'))

    #RP/Gr Orchid Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[4, 0], rgb_CIS[6, 0]]), 'Tender'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[12, 6], rgb_CIS[5, 0]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[4, 8], rgb_CIS[6, 8]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[12, 6], rgb_CIS[5, 9]]), 'Mild'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[7, 0], rgb_CIS[5, 8]]), 'Emotional'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[12, 7], rgb_CIS[5, 8]]), 'Subtle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[1, 0], rgb_CIS[8, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[6, 9], rgb_CIS[5, 9]]), 'Graceful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[7, 9], rgb_CIS[5, 8], rgb_CIS[7, 7]]), 'Chic'))

    #RP/Dl Old Mauve
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[7, 8], rgb_CIS[5, 9]]), 'Sleek'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[6, 2], rgb_CIS[7, 5]]), 'Interesting'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[12, 8], rgb_CIS[5, 8]]), 'Cultured'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[6, 9], rgb_CIS[8, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[1, 2], rgb_CIS[8, 5]]), 'Complex'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[11, 1], rgb_CIS[7, 2]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[1, 0], rgb_CIS[9, 9]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[9, 8], rgb_CIS[1, 2]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[8, 9], rgb_CIS[8, 1], rgb_CIS[11, 6]]), 'Classic'))

    #RP/Dp Wine
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[1, 0], rgb_CIS[8, 2]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[1, 2], rgb_CIS[10, 8]]), 'Extravagant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[9, 2], rgb_CIS[9, 8]]), 'Gorgeous'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[8, 1], rgb_CIS[10, 8]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[1, 2], rgb_CIS[9, 4]]), 'Ethnic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[8, 5], rgb_CIS[9, 2]]), 'Decorative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[1, 9], rgb_CIS[9, 8]]), 'Alluring'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[9, 9], rgb_CIS[1, 2], rgb_CIS[12, 0]]), 'Luxurious'))

    #RP/Dk Red Grape
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[8, 2], rgb_CIS[1, 2]]), 'Substantial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[10, 1], rgb_CIS[8, 0]]), 'Mellow'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[11, 2], rgb_CIS[8, 2]]), 'Tasteful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[10, 5], rgb_CIS[1, 2]]), 'Elaborate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[8, 1], rgb_CIS[10, 8]]), 'Classic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[12, 0], rgb_CIS[7, 7]]), 'Majestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[1, 0], rgb_CIS[11, 9]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[8, 2], rgb_CIS[10, 3]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[10, 9], rgb_CIS[12, 1], rgb_CIS[11, 6]]), 'Strong & Robust'))

    #RP/Dgr Taupe Brown
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[1, 0], rgb_CIS[1, 2]]), 'Luxurious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[10, 1], rgb_CIS[8, 2]]), 'Rustic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[4, 6], rgb_CIS[10, 7]]), 'Intrepid'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[10, 9], rgb_CIS[1, 0]]), 'Mature'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[12, 0], rgb_CIS[10, 2]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[12, 5], rgb_CIS[11, 8]]), 'Majestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[10, 1], rgb_CIS[11, 4]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[12, 3], rgb_CIS[11, 7]]), 'Serious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([rgb_CIS[11, 9], rgb_CIS[12, 4], rgb_CIS[12, 0]]), 'Solemn'))

    #N1.5 Black
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [231,47,39], [255, 200, 8]]), 'Bold'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [44,77,143], [117,173,169]]), 'Stylish'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [194,222,242], [4,148,87]]), 'Modern'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [238,113,25], [231,47,39]]), 'Fiery'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [46,20,141], [255,200,8]]), 'Intense'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [244,244,244], [3,86,155]]), 'Sharp'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [204,63,92], [4,148,87]]), 'Striking'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [85,55,43], [158,128,110]]), 'Strong & Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[10,10,10], [53,109,98], [180,180,180]]), 'Metallic'))

    #N2 Charcoal Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [151,150,139], [180,180,180]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [206,206,206], [117,173,169]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [236,236,236], [133,154,153]]), 'Formal'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [115,63,44], [167,100,67]]), 'Old-fashioned'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [75,63,45], [144,135,96]]), 'Sturdy'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [5,57,107], [130,154,153]]), 'Sublime'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [79,46,43], [172,36,48]]), 'Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [115,63,44], [25,62,63]]), 'Heavy & Deep'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[38,38,38], [25,62,63], [130,154,145]]), 'Authoritative'))


    #N3 Smoke Gray II
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [255,203,88], [130,154,145]]), 'Salty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [126,126,126], [165,184,199]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [130,154,145], [127,175,166]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [75,63,45], [148,133,105]]), 'Bitter'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [126,126,126], [158,128,110]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [152,152,152], [16,76,84]]), 'Authoritative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [148,133,105], [115,63,44]]), 'Conservative'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [103,91,44], [34,62,51]]), 'Quiet & Sophisticated'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[60,60,60], [133,154,153], [25,62,63]]), 'Solemn'))

    #N4 Smoke Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [206,206,206], [144,133,105]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [160,147,131], [184,190,189]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [151,150,139], [127,175,166]]), 'Urbane'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [158,128,110], [75,63,45]]), 'Diligent'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [38,38,38], [148,133,105]]), 'Subtle & Mysterious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [236,236,236], [25,62,63]]), 'Formal'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [79,46,43], [10,10,10]]), 'Serious'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [29,60,47], [165,184,199]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[86,86,86], [130,154,145], [38,38,38]]), 'Precise'))


    #N5 Medium Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [180,180,180], [144,135,96]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [236,236,236], [122,165,123]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [18,83,65], [236,236,236]]), 'Earnest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [38,38,38], [40,57,103]]), 'Majestic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [130,154,145], [25,62,63]]), 'Solemn'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [10,10,10], [8,87,107]]), 'Exact'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [25,62,63], [115,63,44]]), 'Sound'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [75,63,45], [38,38,38]]), 'Strong & Robust'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[126,126,126], [16,76,84], [10,10,10]]), 'Dignified'))


    #N6 Medium Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [144,135,96], [233,227,143]]), 'Simple & Appealing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [206,206,206], [158,128,110]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [203,215,232], [16,76,84]]), 'Intellectual'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [148,133,105], [139,117,65]]), 'Provincial'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [144,135,96], [79,46,43]]), 'Modest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [25,62,63], [38,38,38]]), 'Solemn'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [44,60,49], [24,89,63]]), 'Masculine'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [8,87,107], [115,63,44]]), 'Formal'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[152,152,152], [10,10,10], [151,150,139]]), 'Sublime'))

    #N7 Silver gray II
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [151,150,139], [53,52,48]]), 'Elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [179,202,157], [233,227,143]]), 'Tranquil'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [233,227,143], [184,190,189]]), 'Subtle'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [151,150,139], [130,154,145]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [148,133,105], [255,203,88]]), 'Simple & Appealing'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [25,62,63], [8,87,107]]), 'Earnest'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [144,135,96], [148,133,105]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [115,63,44], [139,117,65]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[180,180,180], [27,86,49], [10,10,10]]), 'Metallic'))

    #N8 Silver gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [158,128,110], [133,154,153]]), 'Chic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [151,150,139], [138,166,187]]), 'Refined'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [53,109,98], [130,154,145]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [180,180,180], [158,128,110]]), 'Simple, quiet & elegant'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [165,184,199], [144,135,96]]), 'Sober'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [18,83,65], [122,165,123]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [24,89,63], [130,154,145]]), 'Distinguished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [6,113,148], [16,76,84]]), 'Rational'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[206,206,206], [4,148,87], [38,38,38]]), 'Modern'))

    #N9 Pearl Gray
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [151,150,139], [206,185,179]]), 'Cultured'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [197,188,213], [165,184,199]]), 'Pure & Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [122,165,123], [133,154,153]]), 'Noble'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [171,131,115], [184,190,189]]), 'Sedate'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [53,109,98], [130,154,145]]), 'Quiet'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [127,175,166], [165,184,199]]), 'Polished'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [148,133,105], [233,227,143]]), 'Dry'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [133,154,153], [20,88,60]]), 'Composed'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[236,236,236], [25,62,63], [126,126,126]]), 'Metallic'))

    #N9.5 White
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [235,219,224], [221,232,207]]), 'Romantic'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [146,198,131], [194,222,242]]), 'Clear'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [59,130,157], [146,198,131]]), 'Crystalline'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [231,47,39], [255,200,8]]), 'Festive'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [138,166,187], [206,206,206]]), 'Simple'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [255,236,79], [88,171,45]]), 'Youthful'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [8,87,107], [231,47,39]]), 'Sporty'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [169,199,35], [59,130,157]]), 'Fresh'))
    kobayashi_colors_new.append(Kobayashi_color(np.array([[244,244,244], [4,148,87], [10,10,10]]), 'Sharp'))

    return kobayashi_colors_new

