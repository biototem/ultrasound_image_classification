from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20


def draw_confusion_matrix(cm, labels, impath=None, is_show=False):
    assert cm.shape[0] == len(labels) and cm.shape[1] == len(labels)
    f = plt.figure(1, clear=True)
    plt.subplot(111)
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap=plt.cm.Blues)
    # need to add 2 line to avoid lack a part of image
    plt.xlim(-0.5, cm.shape[1]+0.5)
    plt.ylim(cm.shape[1]+0.5, -0.5)
    plt.tight_layout()
    if impath is not None:
        f.savefig(impath)
    if is_show:
        plt.show()


if __name__ == '__main__':
    import numpy as np
    cm = np.random.randint(0, 10, [4, 4])
    draw_confusion_matrix(cm, ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
    f = plt.figure(1)
    plt.ion()
    f.show()
    plt.pause(5)
