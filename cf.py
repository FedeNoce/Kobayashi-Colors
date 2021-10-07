import itertools
import numpy as np
import matplotlib.pyplot as plt
cm = np.array([[ 281,   15,   10,    6,   29,    0,   98,    0,    0,   13,   12,    4,    0,    3],
 [  23,  268,    0,    0,  133,    0,   14,   11,    0,    5,   15,    1,    0,    1],
 [   1,   0,  275,   38,    0,    0,   77,    0,    0,    0,    3,    1,    0,   63],
 [   2,    0,   27,  179,    2,    0,   14,    0,    0,    0,    8,    0,    0,    7],
 [  18,   87,    1,    0, 2470,    1,   13,   23,   13,    3,  236,    6,    0,    5],
 [   0,    0,    0,    0,    0,   54,    0,   17,    0,    4,    0,    1,    0,    0],
 [ 132,    9,   57,   19,   41,    0, 1584,    0,    0,    4,   22,   38,    0,  216],
 [   0,   9,    0,    0,   11,   12,    0,  140,    0,    5,    6,    1,    0,    0],
 [   1,    0,    0,    0,   60,    0,    0,    0,   44,    0,   18,    2,    0,    0],
 [   4,    9,    0,    1,    2,    0,    9,    4,    0,   63,    0,    4,    0,    0],
 [  12,   8,    2,    7,  276,    0,   34,    0,    9,    0, 1000,    0,    0,    6],
 [   7,    0,    1,    0,   19,    0,   65,    0,    2,    2,    0,  208,    0,   27],
 [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    3,    0,    0],
 [   1,    0,   49,    5,    3,    0,  198,    0,    0,    0,    5,   20,    0, 1428]])
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, ['CHIC', 'CLASSIC', 'CLEAR', 'COOL-CASUAL', 'DANDY', 'DYNAMIC', 'ELEGANT', 'ETHNIC', 'FORMAL', 'GORGEOUS', 'MODERN', 'NATURAL', 'PRETTY', 'ROMANTIC'])