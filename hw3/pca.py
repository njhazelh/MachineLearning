"""
This script will perform a collection of tests using PCA and a Linear SVM.

The only thing that changes in each run is the number of components
in the PCA transformation.

Once the SVM has been trained on the transformed data, I analyze
the accuracy of the model.

Outputs:
- A figure containing the confusion graphs of each experiment
- A line plot relating the number of PCA components to SVM accuracy
"""

import numpy as np
from sklearn.svm import LinearSVC as SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def measure_accuracy(model, x, y):
    """
    Measure the number of accurate predictions over a body of points.
    :param df: The dataframe containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    guesses = []
    for x_i, y_i in zip(x, y):
        guess = model.predict([x_i])
        accurate += 1 if guess == y_i else 0
        guesses.append(guess)
    return float(accurate), float(len(x)), guesses

def make_confusion_plot(Y, Y_pred, c, ax):
    """
    Render the confusion plot of the data into the axis, ax.
    :param Y: The actual labels
    :param Y_pred: The predicted labels
    :param c: The number of PCA components
    :param ax: The Axis to render the plot into
    """
    ax.set_title("Confusion Matrix K=%d" % c)
    ax.imshow(confusion_matrix(Y, Y_pred))
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('True Label')
    ax.set_ylabel('Predicted Label')

def main():
    train = pd.read_csv('data/digits/train.csv')
    Y = train['label'].values
    X = train.drop('label', axis=1).values
    print("Finished reading training data")

    test = pd.read_csv('data/digits/test.csv')
    Y_test = test['label'].values
    X_test = test.drop('label', axis=1).values
    print("Finished reading testing data")

    points = []
    components = [2, 5, 10, 20, 50, 100, 200]
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3)
    for i, c in enumerate(components):
        print("Running with %d components" % c)
        pca = PCA(n_components=c)
        pca.fit(X)
        print("\tFinished fitting data to PCA")
        tf = pca.transform(X)
        print("\tFinished transforming training data")
        svm = SVC()
        svm.fit(tf, Y)
        print("\tFinished fitting data to SVM")
        accurate, total, guesses = measure_accuracy(svm, pca.transform(X_test), Y_test)
        print("\tFinished testing accuracy: %d/%d = %.4f" % (accurate, total, accurate/total))
        points.append(accurate/total)
        make_confusion_plot(Y_test, guesses, c, fig.add_subplot(gs[i]))
    gs.tight_layout(fig)

    # Plot component-count against accuracy
    plt.figure()
    plt.title('PCA component count vs. Linear SVM Accuracy')
    plt.xlabel('Number of PCA components')
    plt.ylabel('Accuracy (accurate/total)')
    plt.xscale('log')
    plt.xticks(components, components)
    plt.plot(components, points)
    plt.show()

if __name__ == "__main__":
    main()
