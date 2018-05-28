import numpy as np

def sigm(a):
	return 1 / (1 + np.exp(-a))
def sigmDer(a):
	return sigm(a) * (1 - sigm(a))
class Neural:
	def __init__(self, L, D, F, FDer):
		assert L == len(D) - 1 and L == len(F) and L == len(FDer)
		self.l = L
		self.Theta = []
		self.F = F
		for l in range(0, L):
			theta = np.zeros(D[l+1] * (D[l]+1)).reshape(D[l+1], D[l]+1)
			self.Theta.append(theta)

	def update(self, x):
		A = [x]
		a = np.append([1], x).reshape(-1, 1)
		Z = []
		for l in range(0, self.l):
			z = np.dot(self.Theta[l], a)
			Z.append(z)
			a = self.F[l](z)
			A.append(a)
			a = np.append([1], a)
		return Z, A

	def forward0(self, x):
		o = np.dot(self.theta0, x)
		o = np.exp(-o)
		return 1 / (1 + o)
	
	def forward1(self, o):
		return np.dot(self.theta1, o)
		
	def backward1(self, y, yhat, o):
		return (y * (1- yhat) -(1 - y) * yhat) * o

	def backward0(self, y, yhat, o, theta1):
		oo = np.multiply(o, 1 - o)
		oo = np.multiply(oo, theta1)
		oo = np.diag(oo)
		return (y * (1- yhat) - (1 - y) * yhat) * oo * x
	
	def train(self, x, y):
		o = forward0(x)
		yhat = forward1(o)
		theta1 = backward1(y, yhat, o)
		theta0 = backward0(y, yhat, o, theta1)


n = 20
d = 3
x = np.random.rand(n * d).reshape(n, d)
epsilon = np.random.rand(n * d).reshape(n, d) / 100
y = 3 * x[:, 0] + 2 * x[:, 1] + 10 * x[:, 2] + epsilon[:, 0]
y = y.reshape(n, 1)

F = [sigm ,sigm]
FDer = [sigmDer, sigmDer]
model = Neural(2, [3, 2, 1], F, FDer)

Z, A = model.update(x[0, :].reshape(-1, 1))
print Z
print A