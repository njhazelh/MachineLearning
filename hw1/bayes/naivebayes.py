import math
import pandas as pd
import numpy as np

class NaiveBayes:
    """
    Predicts the probability of a single class being true.
    Should only be used on binary class systems where p(C1) = 1 - p(C2)
    """

    def __init__(self, df, label_col, classes):
        """
        Initialize starter values and caches
        :param df: The dataframe containing all the training data.
        :param label_col: A string containing the name of the column that contains the label (value to predict).
        :param classes: A list of the possible labels that the data may have.
        """
        self.df = df
        self.params = []
        self.label_col = label_col
        self.classes = classes
        self.priors = dict()
        size = len(df)
        counts = df[label_col].value_counts()
        for cls in classes:
            if cls in counts:
                self.priors[cls] = float(counts[cls]) / size
            else:
                self.priors[cls] = 0.0
        self.examples = dict()

    def add_param(self, param):
        """
        Add a parameter to the model
        :param param: The parameter to add.  It must have a predict method.  See GaussianParam and CatagoryParam
        """
        self.params.append(param)

    def get_examples(self, cls):
        """
        A helper function to get rows with a specific label.  Makes working with the cache easier.
        :param cls: The class of label that we want to get rows for
        :return: A dataframe containing only rows with the appropriate label.
        """
        if cls not in self.examples:
            self.examples[cls] = self.df[self.df[self.label_col] == cls]
        return self.examples[cls]

    def predict(self, point):
        """
        Predict the best label for the point
        :param point: The point to label
        :return: The label that best matches the point.
        """
        best = (None, None)
        for cls in self.classes:
            examples = self.get_examples(cls)
            p = math.log(self.priors[cls])
            for param in self.params:
                p_i = param.predict(cls, point, examples)
                p += math.log(p_i) if p_i != 0 else -10000
            best = (p, cls) if best[0] is None else max(best, (p, cls), key=lambda x: x[0])
        return best[1]

class GaussianParam:
    """
    GaussianParam is a class that models a normal distribution over continuous variables.
    """

    def __init__(self, name):
        """
        Initialize values and cache.
        :param name: The name of the feature in the dataset.
        """
        self.examples = dict()
        self.name = name
        self.stats = dict()

    def normal_prob(self, mean, std, x):
        """
        A helper function to compute the probability of a value in a normal distribution.
        :param mean: The mean of the population
        :param std: The std of the population
        :param x: The value to calculate the probability for.
        :return: The probability that x would occur given the data.
        """
        return math.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (2 * math.pi * std ** 2) ** .5

    def predict(self, cls, point, examples):
        """
        Calculate p(point[name]|cls) using the Gaussian distribution.
        :param cls: The class we're computing over.
        :param point: The point we want to get the probability of
        :param examples: A dataframe containing all rows with label == cls
        :return: p(point[name]|class)
        """
        value = point[self.name]
        if cls not in self.stats:
            self.stats[cls] = dict()
        if value not in self.stats[cls]:
            examples = examples[self.name]
            self.stats[cls][value] = (examples.mean(), examples.std())
        mean, std = self.stats[cls][value]
        return self.normal_prob(mean, std, value)

class CategoryParam:
    """
    This class represents a parameter while models a probability distribution
    over one or more discrete classes.
    """
    def __init__(self, name):
        """
        Init values and caches.
        :param name: The name of the feature in the dataset
        """
        self.name = name
        self.probs = dict()

    def _prob(self, value, occurences, size):
        """
        A helper function for calculating probability
        :param value: The value we want a probability for
        :param occurences: A mapping of values to occurence counts
        :param size: The number of total items in the population
        :return: The probability the value would be drawn from the population at random
        """
        count = occurences[value] if value in occurences else 0
        return float(count) / size

    def predict(self, cls, point, examples):
        """
        Calculate p(point[name]|cls) using a simple fraction.
        :param cls: The class we're computing over.
        :param point: The point we want to get the probability of
        :param examples: A dataframe containing all rows with label == cls
        :return: p(point[name]|class)
        """
        value = point[self.name]
        if cls not in self.probs:
            self.probs[cls] = dict()
        if value in self.probs[cls]:
            return self.probs[cls][value]
        else:
            examples = examples[self.name]
            self.probs[cls][value] = self._prob(point[self.name], examples.value_counts(), len(examples))
            return self.probs[cls][value]

def clean_data(df):
    """
    Do some data manipulation to make sure everything matches.
    :param df: The dataframe containing all the data.
    :return: The same dataframe with the data cleaned.
    """
    df['income'] = df.income.apply(lambda s: '>50K' in s)
    df['marital-status'] = df['marital-status'].apply(lambda s: s.strip().lower())
    df['workclass'] = df['workclass'].apply(lambda s: s.strip().lower())
    df['education'] = df['education'].apply(lambda s: s.strip().lower())
    df['occupation'] = df['occupation'].apply(lambda s: s.strip().lower())
    df['relationship'] = df['relationship'].apply(lambda s: s.strip().lower())
    df['race'] = df['race'].apply(lambda s: s.strip().lower())
    df['sex'] = df['sex'].apply(lambda s: s.strip().lower())
    df['native-country'] = df['native-country'].apply(lambda s: s.strip().lower())
    return df

def measure_accuracy(df, model):
    """
    Measure the number of accurate predictions over a body of points.
    :param df: The dataframe containing the points to test against.
    :param model: The model to test the points against.
    :return: The number of accurate guesses, and the number of total guesses.
    """
    accurate = 0
    total = len(df)
    count = 0
    for index, adult in df.iterrows():
        count += 1
        if count % 1000 == 0:
            print("%d/%d" % (count, total))
        accurate += 1 if model.predict(adult) == adult.income else 0
    return accurate, total

def main():
    df = pd.read_csv('resources/adult.data')
    df.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = clean_data(df)
    df.reindex(np.random.permutation(len(df)))
    df_training = df.head(30000)
    df_validation = df.tail(2560)

    df_test = pd.read_csv('resources/adult.test')
    df_test.columns = ['age', 'workclass','fnlwgt', 'education', 'education-num',
                  'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country', 'income']
    df_test = clean_data(df_test)

    model = NaiveBayes(df_training, 'income', [True, False])
    model.add_param(CategoryParam('marital-status'))
    model.add_param(CategoryParam('relationship'))
    model.add_param(CategoryParam('native-country'))
    model.add_param(CategoryParam('workclass'))
    model.add_param(CategoryParam('race'))
    model.add_param(CategoryParam('sex'))
    model.add_param(CategoryParam('education'))
    model.add_param(CategoryParam('occupation'))
    model.add_param(GaussianParam('age'))
    model.add_param(GaussianParam('fnlwgt'))
    model.add_param(GaussianParam('education-num'))
    model.add_param(GaussianParam('capital-gain'))
    model.add_param(GaussianParam('capital-loss'))
    model.add_param(GaussianParam('hours-per-week'))

    accurate, total = measure_accuracy(df_training, model)
    print("Training accuracy: %d/%d = %.2f%%" % (accurate, len(df_training), accurate / len(df_training) * 100))
    accurate, total = measure_accuracy(df_validation, model)
    print("Validation accuracy: %d/%d = %.2f%%" % (accurate, len(df_validation), accurate / len(df_validation) * 100))
    accurate, total = measure_accuracy(df_test, model)
    print("Test accuracy: %d/%d = %.2f%%" % (accurate, len(df_test), accurate / len(df_test) * 100))

if __name__ == "__main__":
    main()
