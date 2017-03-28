import random
from pprint import pprint
import matplotlib.pyplot as plt

def dist(a, b):
    """
    Find the euclidean distance between a and b
    :param a: A point, (x1, y1)
    :param b: A point, (x2, y2) 
    :return: The euclidean distance between a and b.
    """
    return ((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2) ** 0.5

def avg(points):
    """
    Find the mean of each dimension
    :param points: The points to average, (x1, y1)...(xn, yn)
    :return: The average of each dimension (x_mean, y_mean)
    """
    x_mean = sum([p[0] for p in points]) / len(points)
    y_mean = sum([p[1] for p in points]) / len(points)
    return x_mean, y_mean

def main():
    """
    Perform K-means on the problem data
    """
    green = [(25,125), (29,97), (44, 105), (35, 63), (42, 57),
             (23, 40), (33, 22), (55, 63), (55, 20), (64, 37)]
    yellow = [(28, 145), (50, 130), (65, 140), (55, 118), (38, 115),
              (50, 90), (63, 88), (43, 83), (50, 60), (50, 30)]

    # I'm making an assumption here that both the green and yellow
    # points are part of the data, and that we just randomly chose
    # the green points as the initial centroids.
    # This makes sense, because otherwise, we're trying to fit k=10
    # to ten points, which is kinda pointless.
    data = green + yellow
    last_centroids = []
    centroids = green

    # move the centroids until they don't change
    while last_centroids != centroids:
        # Account for centroids being pruned to preserve K.
        while len(centroids) < len(green):
            centroids.append((random.choice(data)))
        clusters = {}

        for point in data:
            closest = min(centroids, key=lambda c: dist(c, point))
            if closest not in clusters:
                clusters[closest] = [point]
            else:
                clusters[closest] += [point]

        last_centroids = centroids
        centroids = [avg(points) for points in clusters.values()]
        pprint(clusters)
        pprint(centroids)

        plt.figure()

        # plot the green points
        x = [p[0] for p in green]
        y = [p[1] for p in green]
        plt.scatter(x, y, color='green')

        # plot the yellow points
        x = [p[0] for p in yellow]
        y = [p[1] for p in yellow]
        plt.scatter(x, y, color='yellow')

        # plot the centroids
        centroid_x = [c[0] for c in last_centroids]
        centroid_y = [c[1] for c in last_centroids]
        plt.scatter(centroid_x, centroid_y, color='red', marker='x')

        # draw lines from points to their centroids
        for centroid, points in clusters.items():
            for x, y in points:
                plt.plot([centroid[0], x], [centroid[1], y], '#000000aa')
    plt.show()

if __name__ == "__main__":
    main()
