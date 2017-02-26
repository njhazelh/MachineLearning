import math


def entropy(parts):
    total = sum(parts)
    return 0.0 if any((p == total for p in parts)) else -sum([p/total*math.log(p/total, 2) for p in parts])

def information_gain(before, *after):
    total = sum(before)
    return entropy(before) - sum(sum(x)/total * entropy(x) for x in after)

def main():
    print(information_gain([9, 7], [8, 5], [1, 2]))
    print(information_gain([9, 7], [3, 5], [6, 2]))
    print(information_gain([9, 7], [6, 6], [3, 1]))

if __name__ == "__main__":
    main()
