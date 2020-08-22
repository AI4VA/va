import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
import argparse

def draw_tsne(dataFile, labelFile=None):
    X_embedded = np.load(dataFile).astype(float)
    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    plt.figure(figsize=(16, 10))
    dataPoints = dict()
    dataPoints['x'] = x
    dataPoints['y'] = y
    if labelFile is not None:
        labels = np.load(labelFile).astype(int)
        dataPoints['label'] = labels[:]
        sns.scatterplot(
            x="x", y="y",
            hue='label',
            palette=["green", "red"],
            data=dataPoints,
            legend="full",
            alpha=0.3
        )
    else:
        labels = np.array(range(len(x)))
        dataPoints['label'] = labels[:]
        sns.scatterplot(
            x="x", y="y",
            hue='label',
            palette=["green", "red"],
            data=dataPoints,
            legend="full",
            alpha=0.3
        )
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="data points file")
    parser.add_argument("--label", help="data lable file")
    args = parser.parse_args()
    if args.label:
        draw_tsne(args.data, args.label)
    else:
        draw_tsne(args.data)


if __name__ == '__main__':
    main()