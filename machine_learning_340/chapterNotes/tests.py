#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
def calcStat(X):
    size = X.shape[0]
    x_bar = np.sum(X, axis=0) / size
    S = np.outer(X[0, :], X[0, :])
    for i in range(1, size):
        S = S + np.outer(X[i, :], X[i, :])
    S = S / size - np.outer(x_bar, x_bar)
    return x_bar, S
            
mu1 = np.array([0, 1])
sigma1 = np.array([[1, .5], [.5, 1]])
mu2 = np.array([12, 2])
sigma2 = np.array([[3, -3], [-3, 20]])
size = 100

X1 = np.random.multivariate_normal(mu1, sigma1, size)
x_bar1, S1 = calcStat(X1)
print("mean of X1 = ",  x_bar1)
print("S1 = ")
print(S1)

X2 = np.random.multivariate_normal(mu2, sigma2, size)
x_bar2, S2 = calcStat(X2)
print("mean of X2 = ",  x_bar2)
print("S2 = ")
print(S2)

fig = plt.figure(figsize=(10, 30))
ax = fig.add_subplot(111)
ax.scatter(X1[:, 0], X1[:, 1])
ax.scatter(X2[:, 0], X2[:, 1])
plt.show()
