"""
This script will map the digits dataset to two dimensions using
PCA, and then render the transformed datapoints to a scatterplot.
"""

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def main():
    train = pd.read_csv('data/digits/train.csv')[:10000]
    Y = train['label'].values
    X = train.drop('label', axis=1).values

    pca = PCA(n_components=2)
    pca.fit(X)
    tf = pca.transform(X)
    for n in range(10):
        x = tf[Y == n]
        plt.scatter(x[:,0], x[:,1], label=str(n))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
