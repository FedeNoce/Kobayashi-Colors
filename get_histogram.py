import csv
import matplotlib.pyplot as plt
import pandas as pd

labels = ['Charming', 'Agreeable to  the touch', 'Innnocent', 'Soft', 'Sweet & Dreamy', 'Amiable', \
          'Supple', 'Dreamy', 'Romantic', 'Light', 'Neat', 'Fresh & Young', 'Clear', 'Pure', 'Clean', \
          'Crystalline', 'Refreshing', 'Simple', 'Pure & Simple', 'Clean & Fresh', 'Youthful', 'Steady', \
          'Young', 'Speedy', 'Agile', 'Western', 'Sporty', 'Smart', 'Polished', 'Urban', 'Composite', 'Progressive', \
          'Distinguished', 'Intellectual', 'Modern', 'Cultivated', 'Precise', 'Exact', 'Rational', 'Metallic', \
          'Sublime', 'Earnest', 'Proper', 'Composed', 'Masculine', 'Diligent', 'Subtle & Mysterious', 'Quiet & Sophisticated', \
          'Eminent', 'Bitter', 'Placid', 'Aristocratic', 'Dapper', 'Precious', 'Formal', 'Solemn', 'Pratical', \
          'Sound', 'Majestic', 'Heavy & Deep', 'Strong & Robust', 'Serious', 'Dignified', 'Quiet', 'Chic', 'Noble & Elegant', \
          'Japanese', 'Modest', 'Simple, quiet & elegant', 'Sober', 'Stilish', 'Provincial', 'Rustic', 'Tasteful', 'Complex', \
          'Mellow', 'Old-fashioned', 'Classic', 'Traditional', 'Conservative', 'Elaborate', 'Sturdy', 'Feminine', \
          'Cultured', 'Delicate', 'Tender', 'Natural', 'Emotional', 'Dry', 'Simple & Appealing', 'Sleek', 'Pure & Elegant', \
          'Sedate', 'Noble', 'Fashionable', 'Refined', 'Subtle', 'Interesting', 'Mysterious', 'Graceful', 'Elegant', 'Gentle & Elegant', \
          'Brillant', 'Calm', 'Nostalgic', 'Delicious', 'Mild', 'Open', 'Domestic', 'Smooth', 'Healty', 'Restful', 'Sweet-sour', \
          'Free', 'Pleasant', 'Generous', 'Intimate', 'Gentle', 'Sunny', 'Wholesome', 'Citrus', 'Peaceful', 'Tranquil', 'Fresh', \
          'Plain', 'Friendly', 'Lighthearted', 'Fascinating', 'Substantial', 'Glossy', 'Alluring', 'Aromatic', 'Mature', 'Extravagant', \
          'Gorgeous', 'Luxurious', 'Decorative', 'Lively', 'Hot', 'Provocative', 'Vigorous', 'Dynamic', 'Forceful', 'Bold', 'Dynamic & Active', 'Active', \
          'Fiery', 'Striking', 'Intense', 'Cheerful', 'Happy', 'Enjoyable', 'Festive', 'Bright', 'Dazzling', 'Merry', \
          'Amusing', 'Casual', 'Flamboyant', 'Showy', 'Vivid', 'Tropical', 'Colorful', 'Pretty', 'Cute', 'Childlike', 'Sweet', 'Ethnic', \
          'Untamed', 'Fruitful', 'Wild', 'Robust', 'Autoritative', 'Deep', 'Grand', 'Intrepid', 'Salty', 'Aqueous', 'Pastoral', 'Artistic', \
          'Dewy']
print(len(labels))
for label in labels:
    i = 0
    with open('/data/fnocentini/data/good_outfits_kmeans_5_subtraction.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if label == row[3]:
                i = i + 1
        with open('/data/fnocentini/data/histogram_k_means_5_subclass.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([label, i])


data = pd.read_csv('/data/fnocentini/data/histogram_k_means_5_subclass.csv', sep=',',header=None, index_col =0)

data.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Label')
plt.title('Title')

plt.show()

labels_ = ['ROMANTIC', 'CLEAR', 'COOL-CASUAL', 'NATURAL', 'ELEGANT', 'FORMAL', 'MODERN', 'CHIC', 'DANDY', 'CLASSIC', 'GORGEOUS', \
           'ETHNIC', 'DYNAMIC', 'CASUAL', 'PRETTY', 'NO CLASS DEFINED']

print(len(labels_))
for label in labels_:
    i = 0
    with open('/data/fnocentini/data/good_outfits_kmeans_5_subtraction.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if label == row[2]:
                i = i + 1
        with open('/data/fnocentini/data/histogram_k_means_5_class.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([label, i])


data = pd.read_csv('/data/fnocentini/data/histogram_class.csv', sep=',',header=None, index_col =0)

data.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Class')
plt.title('CLASS HISTOGRAM')

plt.show()