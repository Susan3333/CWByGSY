import math

import numpy as np
import matplotlib.pyplot as plt

func_name = "sigmoid"


def relu(x):
    if x > 0:
        return x
    else:
        return 0


def leakyrelu(x):
    if x > 0:
        return x
    else:
        return x * 0.1


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def func4():
    x = np.arange(-5, 5, 0.02)
    y = []
    for i in x:
        yi = sigmoid(i)
        y.append(yi)
    plt.xlabel('x')
    plt.ylabel('y ' + func_name + '(x)')
    plt.title(func_name)
    plt.plot(x, y)
    plt.show()


def demo():
    import random
    a = 0.1
    x = random.randint(1, 10)
    y = x * x + 2
    index = 1
    while index < 100 and abs(y - 2) > 0.01:
        y = x * x + 2
        print("batch={}  x={}  y={}".format(index, x, y))
        x = x - 2 * x * a
        index += 1


demo()
