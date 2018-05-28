import numpy as np
n = 20
d = 3
x = np.random.rand(n * d).reshape(n, d)
epsilon = np.random.rand(n * d).reshape(n, d) / 100
y = 3 * x[:, 0] + 2 * x[:, 1] + 10 * x[:, 2] + epsilon[:, 0]
y = y.reshape(n, 1)
def newtonLinRegDF(X, y):
	inv = np.linalg.inv(np.dot(X.T, X))
	theta = np.dot(np.dot(inv, X.T), y)
	return theta

def gradientDescentLinReg(X, y, eta, iterations, theta = 0):
	n, d = X.shape
	try:
		if theta.shape[0] != d or theta.shape[1] != 1:
			theta = np.zeros(d).reshape(d, 1)
	except:
		theta = np.zeros(d).reshape(d, 1)
	xtx = np.dot(X.T, X)
	xty = np.dot(X.T, y)
	for i in range(0, iterations):
		exp = np.dot(xtx, theta)
		exp = exp - xty
		theta = theta - eta * exp
	return theta.T 
def onlineGradientDescent(theta, x, yk):
	temp = yk - np.dot(x.T, theta)
	return theta + temp * x

def onlineTest(X, y):
	n, d = X.shape
	theta = np.zeros(d).reshape(d, 1)
	for i in range(0, n):
		theta = onlineGradientDescent(theta, X[i, :].reshape(-1, 1), y[i])
	return theta

print newtonLinRegDF(x, y)
theta = onlineTest(x, y)
print theta
print gradientDescentLinReg(x, y, .01, 1000, theta)

