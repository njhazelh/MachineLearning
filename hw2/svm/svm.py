import random

import numpy as np
import pandas as pd

class SVM:

    def __init__(self, kernel, trainer):
        self.kernel = kernel
        self.support_vectors = None
        self.sv_alphas = None
        self.bias = 0
        self.trainer = trainer

    def train(self, data, *trainer_args):
        #data = np.c_[np.ones(len(data)), data]
        np.random.shuffle(data)
        self.support_vectors, self.sv_alphas, self.bias = self.trainer.train(self.kernel, data, *trainer_args)

    def predict(self, point):
        assert self.support_vectors is not None
        assert self.sv_alphas is not None

        prediction = self.bias
        for m, sv in zip(self.sv_alphas, self.support_vectors):
            prediction += m * sv[-1] * self.kernel(sv[:-1], point)
        return np.sign(prediction)


class SMOTrainer:
    def train(self, kernel, data, max_iterations, tolerance, C):
        print("training")
        sample_count = len(data)
        alphas = np.zeros(sample_count)
        bias = 0
        passes = 0
        iterations = 0

        k = np.zeros((sample_count, sample_count))
        for i in range(sample_count):
            for j in range(sample_count):
                k[i, j] = kernel(data[i,:-1], data[j,:-1])

        while passes < max_iterations:
            iterations += 1
            print(iterations, passes)
            print(bias, sum(alphas))
            num_changed_alphas = 0
            for i in range(sample_count):
                point_i = data[i]
                y_i = point_i[-1]
                x_i = point_i[:-1]
                error_i = self.calc_error(k, i, y_i, alphas, data, bias)
                if (y_i * error_i < -tolerance and alphas[i] < C) or (y_i * error_i > tolerance and alphas[i] > 0):
                    j = self.random_int_excluding_i(len(data), i)
                    point_j = data[j]
                    y_j = point_j[-1]
                    x_j = point_j[:-1]
                    error_j = self.calc_error(k, j, y_j, alphas, data, bias)
                    old_alpha_i, old_alpha_j = alphas[i], alphas[j]
                    l = self.calc_l(alphas, C, i, j, y_i, y_j)
                    h = self.calc_h(alphas, C, i, j, y_i, y_j)
                    if l == h:
                        #print("l == h")
                        continue
                    n = self.calc_n(x_i, x_j)
                    if n >= 0:
                        #print("n >= 0")
                        continue

                    new_alpha_j = old_alpha_j - y_j * (error_i - error_j) / n
                    if new_alpha_j > h:
                        new_alpha_j = h
                    elif new_alpha_j < l:
                        new_alpha_j = l
                    alphas[j] = new_alpha_j

                    if abs(new_alpha_j - old_alpha_j) < 1e-5:
                        #print("abs(new_alpha_j - old_alpha_j) < 1e-5")
                        continue
                    new_alpha_i = old_alpha_i + y_i * y_j * (old_alpha_j - new_alpha_j)
                    alphas[i] = new_alpha_i

                    b1 = bias - error_i - y_i * (new_alpha_i - old_alpha_i) * np.dot(x_i, x_i) - y_j * (new_alpha_j - old_alpha_j) * np.dot(x_i, x_j)
                    b2 = bias - error_j - y_i * (new_alpha_i - old_alpha_i) * np.dot(x_i, x_j) - y_j * (new_alpha_j - old_alpha_j) * np.dot(x_j, x_j)
                    if 0 < new_alpha_i < C:
                        bias = b1
                    elif 0 < new_alpha_j < C:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2

                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        print(alphas)
        # TODO extract support vectors
        sv = alphas > 1e-5
        return data[sv], alphas[sv], bias

    def calc_l(self, alphas, C, i, j, y_i, y_j):
        if y_i == y_j:
            return max(0, alphas[i] + alphas[j] - C)
        else:
            return max(0, alphas[j] - alphas[i])

    def calc_h(self, alphas, C, i, j, y_i, y_j):
        if y_i == y_j:
            return min(C, alphas[i] + alphas[j])
        else:
            return min(C, C + alphas[j] - alphas[i])

    def calc_n(self, x_i, x_j):
        return 2 * np.dot(x_i, x_j) - np.dot(x_i, x_i) - np.dot(x_j, x_j)

    def calc_error(self, k, idx, y, alphas, data, bias):
        y_pred = bias
        for i in range(len(data)):
            point = data[i]
            y_pred += alphas[i] * point[-1] * k[i, idx]
        return y_pred - y

    def random_int_excluding_i(self, max_val, i):
        j = random.randrange(max_val - 1)
        return max_val - 1 if j == i else j



def measure_accuracy(df, model):
    """
    Measure the number of accurate predictions over a body of points.
    :param df: The dataframe containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    for row in df:
        accurate += 1 if model.predict(row[:-1]) == row[-1] else 0
    return accurate, len(df)

def main():
    df_training = pd.read_csv("../data/digits/train.csv").values
    df_testing = pd.read_csv("../data/digits/test.csv").values
    #df = np.genfromtxt("../data/perceptron/percep1.txt", delimiter="\t")
    #df = np.genfromtxt("../data/perceptron/percep2.txt", delimiter="\t")
    # df_training = df[:900]
    # df_testing = df[-100:]
    #np.random.shuffle(df)

    kernel = lambda a, b: np.dot(a, b)
    #kernel = lambda a, b: np.exp(-np.dot(a-b, a-b))
    model = SVM(kernel, SMOTrainer())
    model.train(df_training, 1, 0.2, 2)

    accurate, total = measure_accuracy(df_testing, model)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()
