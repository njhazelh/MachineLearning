
import numpy as np

class PrimalPerceptron:
    """
    A perceptron in the primal form.  It can only identify linear patterns in data.
    """

    def __init__(self):
        """
        Init the instance
        """
        self.weights = None
        self.bias = 0

    @property
    def normalized_weights(self):
        """
        :return: The normalized weights of the function.
        """
        return self.weights / -self.bias

    def train(self, data):
        """
        Train the model on the data
        :param data: A numpy array containing a series of datapoints,
            where the last item in each row is the label {-1, 1}
        :return: The number of iterations over the data,
            the number of updates performed
        """
        self.weights = np.zeros(len(data[0]) - 1)
        self.bias = 0
        iterations = 0
        updates = 0
        while True:
            iterations += 1
            new_updates = 0
            for point in data:
                val = (np.dot(point[:-1], self.weights) + self.bias) * point[-1]
                if val <= 0:
                    new_updates += 1
                    updates += 1
                    self.weights += point[-1] * point[:-1]
                    self.bias += point[-1]
            print("Iterations: %d" % iterations)
            print("New Updates: %d\n" % new_updates)
            if new_updates == 0:
                break
        return iterations, updates

    def predict(self, point):
        """
        Predict the class {-1,1} of the given point
        :param point: The point to predict
        :return: The class of the point according to the model.
        """
        if np.dot(point, self.weights) + self.bias >= 0:
            return 1
        else:
            return -1

def measure_accuracy(data, model):
    """
    Measure the number of accurate predictions over a body of points.
    :param data: The numpy array containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    total = len(data)
    for row in data:
        accurate += 1 if model.predict(row[:-1]) == row[-1] else 0
    return accurate, total

def main():
    df = np.genfromtxt("../data/perceptron/percep1.txt", delimiter="\t")
    df_training = df[:900]
    df_testing = df[-100:]

    model = PrimalPerceptron()
    iterations, updates = model.train(df_training)
    print("Model trained: %d iterations, %d updates" % (iterations, updates))
    print("Weights: %s" % list(model.weights))
    print("Normalized Weights: %s" % list(model.normalized_weights))
    print("Bias: %d" % model.bias)

    accurate, total = measure_accuracy(df_testing, model)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()
