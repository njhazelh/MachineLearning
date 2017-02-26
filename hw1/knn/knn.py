import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

STEP = .1

def dist(xy, p2):
    return math.sqrt((xy[0] - p2.x1) ** 2 + (xy[1] - p2.x2) ** 2)

def predict(xy, df, k):
    df['dist'] = df.apply(lambda row: dist(xy, row), axis=1)
    closest = df.nsmallest(k, 'dist')
    return closest[closest.label == 1].label.count() / k

def measure_accuracy(test_df, model_df, k):
    count = len(test_df)
    wrong = 0
    for pt in test_df.values:
        prediction = predict(pt[1:], model_df, k)
        if abs(prediction - pt[0]) > 0.5:
            wrong += 1
    return count - wrong, count

def main():
    train_df = pd.read_csv("resources/train.data")
    test_df = pd.read_csv("resources/test.data")

    max_k = 12
    ks = np.array(range(1, max_k + 1))
    train_accuracies = []
    test_accuracies = []
    for k in ks:
        print("train-%d" % k)
        correct, total = measure_accuracy(train_df, train_df, k)
        train_accuracies.append(correct/total)
    for k in ks:
        print("test-%d" % k)
        correct, total = measure_accuracy(test_df, train_df, k)
        test_accuracies.append(correct/total)

    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)

    bar_width = .25
    fig, ax = plt.subplots()
    bars1 = ax.bar(ks, train_accuracies, bar_width, alpha=.75, label='testing')
    bars2 = ax.bar(ks + bar_width, test_accuracies, bar_width, alpha=.75, label='testing')
    ax.set_xticks(ks + bar_width / 2)
    ax.set_xticklabels(ks)
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy (correct/total)')
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 0.7, '%.3f' % h,
                ha='center', va='bottom', rotation='vertical')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,h * 0.7, '%.3f' % h,
                ha='center', va='bottom', rotation='vertical')
    plt.title('Accuracy of KNN vs value of K')
    plt.legend(['Testing', 'Training'], loc='upper left')
    plt.grid()
    plt.show()

    xx, yy = np.meshgrid(
        np.arange(train_df.x1.min(), train_df.x1.max(), STEP),
        np.arange(train_df.x2.min(), train_df.x2.max(), STEP)
    )
    z1 = np.array([predict(xy, train_df, 1) for xy in np.c_[xx.ravel(), yy.ravel()]]).reshape(xx.shape)
    z9 = np.array([predict(xy, train_df, 9) for xy in np.c_[xx.ravel(), yy.ravel()]]).reshape(xx.shape)

    training_ones = train_df[train_df.label == 1]
    training_zeros = train_df[train_df.label == 0]
    test_ones = test_df[test_df.label == 1]
    test_zeros = test_df[test_df.label == 0]

    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, z1, cmap='bwr')
    ax.scatter(training_ones['x1'], training_ones['x2'], label='train==1', c='red', edgecolors='white')
    ax.scatter(training_zeros['x1'], training_zeros['x2'], label='train==0', c='blue', edgecolors='white')
    ax.scatter(test_ones['x1'], test_ones['x2'], label='test==1', c='red', edgecolors='black')
    ax.scatter(test_zeros['x1'], test_zeros['x2'], label='test==0', c='blue', edgecolors='black')
    plt.title('KNN Decision Surface (K=1)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.legend(['train=1', 'train=0', 'test=1', 'test=0'])

    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, z9, cmap='bwr')
    ax.scatter(training_ones['x1'], training_ones['x2'], label='train==1', c='red', edgecolors='white')
    ax.scatter(training_zeros['x1'], training_zeros['x2'], label='train==0', c='blue', edgecolors='white')
    ax.scatter(test_ones['x1'], test_ones['x2'], label='test==1', c='red', edgecolors='black')
    ax.scatter(test_zeros['x1'], test_zeros['x2'], label='test==0', c='blue', edgecolors='black')
    plt.title('KNN Decision Surface (K=9)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.legend(['train=1', 'train=0', 'test=1', 'test=0'])
    plt.show(block=True)


if __name__ == '__main__':
    main()
