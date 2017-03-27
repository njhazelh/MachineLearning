
import numpy as np

class DualPerceptron:
    """
    This is a perceptron that uses the dual form and can
    take a kernel to match non-linear patterns in data.
    """

    def __init__(self, kernel):
        """
        :param kernel: A functions that performs the kernel operation.
        """
        self.kernel = kernel

    def train(self, data):
        """
        Train the model on the data
        :param data: A numpy array containing a series of datapoints,
            where the last item in each row is the label {-1, 1}
        :return: The number of iterations over the data,
            the number of updates performed,
            the number of support vectors
        """
        data = np.c_[np.ones(len(data)), data]
        sample_count = len(data)

        k = np.fromfunction(np.vectorize(lambda i, j: self.kernel(data[i,:-1], data[j,:-1])),
                            (sample_count, sample_count), dtype=np.int)

        self.m = np.zeros(sample_count, dtype=np.float64)
        updates = 0
        iterations = 0
        while True:
            iterations += 1
            new_updates = 0
            for t in range(sample_count):
                v = np.sum(k[:,t] * self.m * data[:,-1])
                if data[t,-1] * v <= 0:
                    new_updates += 1
                    updates += 1
                    self.m[t] += 1
            if new_updates == 0:
                break
            else:
                print("New Updates: %d" % new_updates)

        sv = self.m != 0
        self.m = self.m[sv]
        self.sv = data[sv]
        return iterations, updates, len(self.m)

    def predict(self, point):
        """
        Predict the class {-1, 1} of a datapoint
        :param point: The datapoint to predict
        :return: The class of the datapoint
        """
        point = np.append(np.ones(1), point)
        s = 0
        for m, sv in zip(self.m, self.sv):
            s += m * sv[-1] * self.kernel(sv[:-1], point)
        if s >= 0:
            return 1
        else:
            return -1

    @property
    def weights(self):
        """
        Access The weights of the model as though it were in the primal form
        :return: The weights of the model.
        """
        features = len(self.sv[0]) - 2
        weights = [0] * features
        for m, sv in zip(self.m, self.sv):
            for i in range(features):
                weights[i] += m * sv[i+1] * sv[-1]
        return weights

    @property
    def normalized_weights(self):
        """
        Access the normalized weights of the model: weights / bias
        :return: The normalized weights of the model.
        """
        bias = self.bias
        return [weight / -bias for weight in self.weights]

    @property
    def bias(self):
        """
        Access the bias of the function.
        :return: The bias of the function
        """
        bias = 0
        for m, sv in zip(self.m, self.sv):
            bias += m * sv[0] * sv[-1]
        return bias


def measure_accuracy(data, model):
    """
    Measure the number of accurate predictions over a body of points.
    :param data: The numpy array containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    for row in data:
        accurate += 1 if model.predict(row[:-1]) == row[-1] else 0
    return accurate, len(data)

def main():
    #df = np.genfromtxt("../data/perceptron/percep1.txt", delimiter="\t")
    df = np.genfromtxt("../data/perceptron/percep2.txt", delimiter="\t")
    #np.random.shuffle(df)

    df_training = df[:900]
    df_testing = df[-100:]

    #kernel = lambda a, b: np.dot(a, b)
    kernel = lambda a, b: np.exp(-np.dot(a-b, a-b))
    model = DualPerceptron(kernel)
    iterations, updates, sv_count = model.train(df_training)
    print("Model trained: %d iterations, %d updates" % (iterations, updates))
    print("Weights: %s" % list(model.weights))
    print("Normalized Weights: %s" % list(model.normalized_weights))
    print("Bias: %d" % model.bias)

    accurate, total = measure_accuracy(df_testing, model)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, total, accurate / total * 100))

if __name__ == "__main__":
    main()

