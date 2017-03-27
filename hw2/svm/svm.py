import random

import numpy as np
import pandas as pd

class SVM:
    """
    SVM is a dual form representation of a Support Vector Machine.
    """

    def __init__(self, kernel, trainer):
        """
        Init the Model
        :param kernel: The kernel function to use
        :param trainer: The Training algorithm to use.
        """
        self.kernel = kernel
        self.support_vectors = None
        self.sv_y = None
        self.sv_alphas = None
        self.bias = 0
        self.trainer = trainer

    def train(self, x, y, **trainer_args):
        """
        Train the model on the data
        :param x: The array containing features of the data
        :param y: The array containing labels of the data
        :param trainer_args: The arguments for the trainer
        """
        self.support_vectors,\
        self.sv_y,\
        self.sv_alphas,\
        self.bias = self.trainer.train(self.kernel, x, y, **trainer_args)

    def predict(self, point):
        """
        Predict the class of a point
        :param point: The point to predict
        :return: The predicted class fo the point
        """
        assert self.support_vectors is not None
        assert self.sv_y is not None
        assert self.sv_alphas is not None

        prediction = self.bias
        for m, sv, sv_y in zip(self.sv_alphas, self.support_vectors, self.sv_y):
            prediction += m * sv_y * self.kernel(sv, point)
        return np.sign(prediction)


class SMOTrainer:
    """
    SMO Trainer is class that can find the support vectors for a SVM using the
    SMO algorithm.
    """

    def train(self, kernel, x, y, clean_passes, tolerance, reg_factor):
        """
        Train an SVM model
        :param kernel: The kernel function to use
        :param x: An array containing datapoints
        :param y: An array containing the labels of the data points
        :param clean_passes: The number of clean_passes required to finish training
        :param tolerance: The tolerance of the model
        :param reg_factor: The regularization factor of the model
        :return: x[sv] - The features of the support vectors,
                y[sv] - The labels of the support vectors,
                alphas[sv] - The alphas of the support vectors,
                bias - The bias of the model.
        """
        sample_count = len(x)
        alphas = np.zeros(sample_count)
        bias = 0
        passes = 0
        iterations = 0

        print("Training started on %d samples" % sample_count)

        k = np.fromfunction(np.vectorize(lambda i, j: kernel(x[i, :-1], x[j, :-1])),
                            (sample_count, sample_count), dtype=np.int)

        print("Finished generating kernel matrix")

        while passes < clean_passes:
            iterations += 1
            print("Iterations=%d, clean-passes=%d/%d, bias=%.4f, sum(alphas)=%.4f" \
                % (iterations, passes, clean_passes, bias, sum(alphas)))
            num_changed_alphas = 0
            for i in range(sample_count):
                error_i = self.calc_error(k, i, y, alphas, bias)
                if (y[i] * error_i < -tolerance and alphas[i] < reg_factor) or (y[i] * error_i > tolerance and alphas[i] > 0):
                    j = self.random_int_excluding_i(len(x), i)
                    error_j = self.calc_error(k, j, y, alphas, bias)
                    old_alpha_i, old_alpha_j = alphas[i], alphas[j]

                    l = self.calc_l(alphas, reg_factor, i, j, y)
                    h = self.calc_h(alphas, reg_factor, i, j, y)
                    if l == h: continue

                    n = self.calc_n(x, i, j)
                    if n >= 0: continue

                    new_alpha_j = old_alpha_j - y[j] * (error_i - error_j) / n
                    if new_alpha_j > h:
                        new_alpha_j = h
                    elif new_alpha_j < l:
                        new_alpha_j = l
                    alphas[j] = new_alpha_j

                    if abs(new_alpha_j - old_alpha_j) < 1e-5: continue

                    new_alpha_i = old_alpha_i + y[i] * y[j] * (old_alpha_j - new_alpha_j)
                    alphas[i] = new_alpha_i

                    bias = self.calc_bias(bias,
                                          x, y,
                                          i, j,
                                          error_i, error_j,
                                          new_alpha_i, old_alpha_i,
                                          new_alpha_j, old_alpha_j, reg_factor)

                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        sv = alphas > 1e-5
        return x[sv], y[sv], alphas[sv], bias

    def calc_l(self, alphas, C, i, j, y):
        """
        Calculate l according to the formula
        :param alphas: The array containing alphas
        :param C: The regularization factor
        :param i: An index
        :param j: Another index
        :param y: The array containing labels
        :return: l
        """
        if y[i] == y[j]:
            return max(0, alphas[i] + alphas[j] - C)
        else:
            return max(0, alphas[j] - alphas[i])

    def calc_h(self, alphas, C, i, j, y):
        """
        Calculate h according to the formula
        :param alphas: The array containing alphas
        :param C: The regularization factor
        :param i: An index
        :param j: Another index
        :param y: The array containing labels
        :return: h
        """
        if y[i] == y[j]:
            return min(C, alphas[i] + alphas[j])
        else:
            return min(C, C + alphas[j] - alphas[i])

    def calc_n(self, x, i, j):
        """
        Calculate n according to the formula
        :param x: The array containing all x points
        :param i: An index into x
        :param j: Anothr index into x
        :return: n
        """
        return 2 * np.dot(x[i], x[j]) - np.dot(x[i], x[i]) - np.dot(x[j], x[j])

    def calc_error(self, k, idx, y, alphas, bias):
        """
        Calculate the error between the prediction of a point and it's actual label
        :param k: The matrix of kernels
        :param idx: The index of the point
        :param y: The array containing all labels
        :param alphas: The array containing all alphas
        :param bias: The bias of the model
        :return: The error between y_pred and y_actual.
        """
        return bias + np.sum(alphas * y * k[:, idx]) - y[idx]

    def random_int_excluding_i(self, max_val, i):
        """
        :param max_val: The max_val, exclusive
        :param i: The exlucded value
        :return: a random int < max_val that is not i
        """
        j = random.randrange(max_val - 1)
        return max_val - 1 if j == i else j

    def calc_bias(self, bias, x, y, i, j,
                  error_i, error_j,
                  new_alpha_i, old_alpha_i,
                  new_alpha_j, old_alpha_j, reg_factor):
        """
        Calculate the new bias
        :param bias: The previous bias
        :param x: The numpy array containing all training x points
        :param y: The numpy array containing all training labels
        :param i: The index of a point
        :param j: The index of a different point
        :param error_i: The error between the models prediction of point i and the actual
        :param error_j: The error between the models prediction of point j and the actual
        :param new_alpha_i: The new alpha of i
        :param old_alpha_i: The old alpha of i
        :param new_alpha_j: The new alpha of j
        :param old_alpha_j: The old alpha of j
        :param reg_factor: The regularization factor, C
        :return:
        """
        b1 = bias - error_i - y[i] * (new_alpha_i - old_alpha_i) \
                              * np.dot(x[i], x[i]) - y[j] * (new_alpha_j - old_alpha_j) * np.dot(x[i], x[j])
        b2 = bias - error_j - y[i] * (new_alpha_i - old_alpha_i) \
                              * np.dot(x[i], x[j]) - y[j] * (new_alpha_j - old_alpha_j) * np.dot(x[j], x[j])
        if 0 < new_alpha_i < reg_factor:
            return b1
        elif 0 < new_alpha_j < reg_factor:
            return b2
        else:
            return (b1 + b2) / 2



def measure_accuracy(model, x, y):
    """
    Measure the number of accurate predictions over a body of points.
    :param df: The dataframe containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    for x_i, y_i in zip(x, y):
        accurate += 1 if model.predict(x_i) == y_i else 0
    return accurate, len(x)

def main():
    # Read in Training Data
    df_training = pd.read_csv("../data/digits/train.csv").values[:100]
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

    # Create the SVM with linear kernel
    model = SVM(lambda a, b: np.dot(a, b), SMOTrainer())
    model.train(x_training, y_training, clean_passes=3, tolerance=.5, reg_factor=.5)

    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()
