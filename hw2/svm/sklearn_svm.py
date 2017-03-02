import numpy as np
import pandas as pd
from sklearn.svm import SVC

def measure_accuracy(model, x, y):
    """
    Measure the number of accurate predictions over a body of points.
    :param df: The dataframe containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    for x_i, y_i in zip(x, y):
        accurate += 1 if model.predict([x_i]) == y_i else 0
    return accurate, len(x)

def main():
    # Read in Training Data
    df_training = pd.read_csv("../data/digits/train.csv").values[:1300]
    x_training = df_training[:,1:]
    y_training = np.array([1 if v == 5 else -1 for v in df_training[:, 0]])

    # Read in Test Data
    df_testing = pd.read_csv("../data/digits/test.csv").values
    x_testing = df_testing[:, 1:]
    y_testing = np.array([1 if v == 5 else -1 for v in df_testing[:, 0]])

    # Filter out features with 0 variance, since they won't help us
    # and they cause div-by-zero errors in the z-score step
    std = np.std(x_training, axis=0)
    x_training = x_training[:, std != 0]
    x_testing = x_testing[:, std != 0]

    # normalize features using z-score: (x - mean) / std
    std = np.std(x_training, axis=0)
    mean = np.std(x_training, axis=0)
    x_training = (x_training - mean) / std
    x_testing = (x_testing - mean) / std

    model = SVC(kernel='linear')
    model.fit(x_training, y_training)
    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("Linear Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

    model = SVC(kernel='poly')
    model.fit(x_training, y_training)
    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("Polynomial Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

    model = SVC(kernel='rbf')
    model.fit(x_training, y_training)
    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("RBF Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

    model = SVC(kernel='sigmoid')
    model.fit(x_training, y_training)
    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("Sigmoid Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))


if __name__ == "__main__":
    main()
