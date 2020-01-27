import numpy as np
from math import pi

# x is a vector of n elements
# the initialization points are in range of: -32.768 < x < 32.768
# the global optimum is in the origin. max_sigma is 12
def Ackley(x):
    n = len(x)
    cosin = [np.cos(2*pi*tmp) for tmp in x]
    power2 = [np.power(tmp ,2) for tmp in x]
    f = -20 * np.exp(-0.2*np.sqrt((1/n * np.sum(power2, axis=0)))) - np.exp((1/n * np.sum(cosin, axis=0))) + 20 + np.exp(1)
    return f

# the global optimum is in origin. and the initial points are in range: -5.12 < x < -5.12. max_sigma is 2
def Rastrigin(x):
    n = len(x)
    f = 10*n
    for tmp in x:
        if tmp < -5.12 or tmp > 5.12:
            f += 10*np.power(tmp, 2)
        else:
            f += np.power(tmp, 2) - 10*np.cos(2*pi*tmp)
    return f

# the initial points are in range: -500 < x < 500. and the global optimum is x_i=420.9687 and y = -408.98287761957596
def Schwefel(x):
    n = len(x)
    f = 418.9829*n
    for tmp in x:
        if tmp < -500 or tmp > 500:
            f += 0.02*np.power(tmp, 2)
        else:
            f += -tmp * np.sin(np.sqrt(np.abs(tmp)))
    return f

# print(Ackley([0]))
# print(Rastrigin([0]))
# print(Schwefel([412]))