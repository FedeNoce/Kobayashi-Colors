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
    with open('/data/fnocentini/good_outfits_subtraction_hsv.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if label == row[3]:
                i = i + 1
        with open('/data/fnocentini/histogram.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([label, i])


data = pd.read_csv('/data/fnocentini/histogram.csv', sep=',',header=None, index_col =0)

data.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Label')
plt.title('Title')

plt.show()