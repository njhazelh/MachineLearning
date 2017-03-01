
import numpy as np
import pandas as pd

class SVM:

    def __init__(self, kernel, trainer):
        self.kernel = kernel
        self.support_vectors = None
        self.sv_alphas = None
        self.trainer = trainer

    def train(self, data, max_iterations):
        data = np.c_[np.ones(len(data)), data]
        np.random.shuffle(data)
        self.support_vectors, self.sv_alphas = self.trainer.train(self.kernel, data, max_iterations)

    def predict(self, point):
        assert self.support_vectors is not None
        assert self.sv_alphas is not None

        prediction = 0
        for m, sv in zip(self.sv_alphas, self.support_vectors):
            prediction += m * sv[-1] * self.kernel(sv[:-1], point)
        return np.sign(prediction)


class SMOTrainer:
    def train(self, kernel, data, max_iterations, tolerance=1, C=10):
        sample_count = len(data)
        alphas = np.zeros(sample_count)
        passes = 0
        bias = 0
        while passes < max_iterations:
            num_changed_alphas = 0
            for i in range(sample_count):
                point_i = data[i]
                y_i = point_i[-1]
                x_i = point_i[:-1]
                error_i = self.calc_error(x_i, y_i, alphas, data, bias)
                if (y_i * error_i < -tolerance and alphas[i] < C) or (y_i * error_i > tolerance and alphas[i] > 0):
                    j = 0 # select random j != i
                    point_j = data[j]
                    y_j = point_j[-1]
                    x_j = point_j[:-1]
                    error_j = self.calc_error(x_j, y_j, alphas, data, bias)
                    old_alpha_i, old_alpha_j = alphas[i], alphas[j]
                    l = self.calc_l(alphas, C, i, j, y_i, y_j)
                    h = self.calc_h(alphas, C, i, j, y_i, y_j)
                    if l == h:
                        continue
                    n = self.calc_n(x_i, x_j)
                    if n >= 0:
                        continue
                    new_alpha_j = 0 # TODO compute new alpha for j
                    if abs(new_alpha_j - old_alpha_j) < 1e-5:
                        continue
                    new_alphas_i = 0  # TODO compute new alpha for i
                    b1 = 0 # TODO compute b1
                    b2 = 0 # TODO compute b2
                    b = 0 # TODO compute b from b1 and b2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        # TODO extract support vectors
        return [], []

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

    def calc_error(self, x, y, alphas, data, bias):
        return 0


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

    print(df_training.describe())

    kernel = lambda a, b: np.dot(a, b)
    #kernel = lambda a, b: np.exp(-np.dot(a-b, a-b))
    model = SVM(kernel, SMOTrainer())
    model.train(df_training, 100)

    accurate, total = measure_accuracy(df_testing, model)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()
