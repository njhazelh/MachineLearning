import numpy as np
import pandas as pd

class MultiLayerPerceptron:
    """
    MultiLayerPercepton is an implementation of MultiLayerPerceptron using backpropagation.
    """

    def __init__(self, layers, learning_rate, accuracy=0.001, max_iters=10000):
        """
        Init the MLP
        :param layers: A list containing the node count for each layer.
        :param learning_rate: The learning rate
        :param accuracy: The max error allowed on the training data
        :param max_iters: The max iterations to be performed if accuracy not met
        """
        self.learning_rate = learning_rate
        self.weight = [np.random.rand(layers[i], layers[i+1]) for i in range(0, len(layers)-1)]
        self.bias = [np.ones(layers[i+1]) for i in range(0, len(layers)-1)]
        self.accuracy = accuracy
        self.max_iters = max_iters

    def train(self, x, y):
        """
        Train the model
        :param x: The training features
        :param y: The labels for the training features
        """
        iterations = 0
        indexes = np.array(range(len(x)))
        error = 2.0
        while .5 * error ** 2 > self.accuracy and iterations < self.max_iters:
            iterations += 1
            if iterations % 10 == 0:
                print("Iteration %d - Error: %.4f" % (iterations, error))
            np.random.shuffle(indexes)
            for i in range(len(x)):
                a = self.forward(x[indexes[i]])
                error = error * .99 + self.backward(a, y[indexes[i]]) * .01

    def backward(self, a, y):
        """
        Perform back-propagation
        :param a: The alphas matrix from forward prop
        :param y: The correct label
        :return: The absolute value of the difference between actual and expected label
        """
        y_pred = np.asscalar(a[-1])
        idx = len(a)-1
        d = (y_pred - y) * self.sigmoid_prime(a[idx])
        deltas =[d]
        idx -= 1

        while idx > 0:
            d = deltas[0].dot(self.weight[idx].T) * self.sigmoid_prime(a[idx])
            deltas.insert(0, d)
            idx -= 1

        idx = len(self.weight)-1
        while idx >= 0:
            self.weight[idx] -= self.learning_rate * a[idx].T.dot(deltas[idx])
            self.bias[idx] -= self.learning_rate * deltas[idx].sum(0) / deltas[idx].shape[0]
            idx -= 1

        return abs(y_pred - y)


    def forward(self, x):
        """
        Perform forward propagation
        :param x: The features to propagate forward
        :return: The matrix of intermediate alphas.
        """
        m = np.atleast_2d(x)
        a = [m]

        for i in range(len(self.weight)):
            m = np.dot(m, self.weight[i]) + self.bias[i]
            m = self.sigmoid(m)
            a.append(m)

        return a

    def predict(self, x):
        """
        Predict the class of a data point.
        :param x: The datapoint to label
        :return: The label of the datapoint
        """
        a = self.forward(x)
        val = np.asscalar(a[-1])
        return 1 if val >= .5 else 0

    def sigmoid(self, val):
        """
        Perform the sigmoid operation
        :param val: The value to operate on
        :return: The result of the operation
        """
        return (np.exp(-val) + 1) ** -1

    def sigmoid_prime(self, val):
        """
        Perform the derivative operation of the sigmoid function
        :param val: The value to operate on
        :return: The result
        """
        return self.sigmoid(val) * (1 - self.sigmoid(val))


def measure_accuracy(model, x, y):
    """
    Measure the number of accurate predictions over a body of points.
    :param model: The model to test the points against.
    :param x: The features of the data
    :param y: The actual labels of the data
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = sum((1 for x_i, y_i in zip(x, y) if model.predict(x_i) == y_i))
    return accurate, len(x)

def main():
    #Read in Training Data
    df_training = pd.read_csv("../data/digits/train.csv").values[:100]
    x_training = df_training[:,1:]
    y_training = np.array([1 if v == 5 else 0 for v in df_training[:, 0]])

    # Read in Test Data
    df_testing = pd.read_csv("../data/digits/test.csv").values
    x_testing = df_testing[:, 1:]
    y_testing = np.array([1 if v == 5 else 0 for v in df_testing[:, 0]])

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

    # x_training = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    # y_training = np.array([1, 1, 0, 0])
    #
    # x_testing = x_training
    # y_testing = y_training

    # Create the SVM with linear kernel
    model = MultiLayerPerceptron([x_training.shape[1], 2, 1],
                                 learning_rate=.1,
                                 accuracy=0.001,
                                 max_iters=100000)
    model.train(x_training, y_training)

    accurate, total = measure_accuracy(model, x_testing, y_testing)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()
