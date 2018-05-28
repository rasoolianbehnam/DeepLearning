import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = W.shape[0]
  C = W.shape[1]
  S = np.zeros((N, C))
  losses = np.zeros(N)
  for i in xrange(N):
      S[i, :] = np.dot(X[i, :], W)
  losses = np.zeros(N)
  for i in xrange(N):
      losses[i] = -np.log(np.exp(S[i, y[i]]) / np.sum(np.exp(S[i, :])))
  #print "losses = ", losses
  loss = np.mean(losses) + reg * np.sum(W*W)
  
  for n in xrange(N):
      b = np.sum(np.exp(S[n, :]))
      for c in xrange(C):
          l = (c == y[n])*1
          a = np.exp(S[n, c])
          dW[:, c] += X[n, :] * (-l + a / b)
  dW = dW / N + reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = W.shape[0]
  C = W.shape[1]
  S = np.dot(X, W)
  #print "S = ", S
  #print "y = ", y
  sums = np.sum(np.exp(S), axis=1).reshape(-1 ,1)
  #print "sum.shape =", sums.shape
  losses = -S.T[y].diagonal() + np.log(sums)
  #print "losses = ", losses
  loss = np.mean(losses) + reg * np.sum(W*W)
  multipliers = np.exp(S) / sums
  multipliers[(xrange(N), y)] -= 1
  dW = X.T.dot(multipliers)
  dW = dW / N + reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

#N = 10
#D = 1000
#C = 10
#X = np.random.rand(N, D)
#y = np.random.rand(N) * 10
#y = y.astype('uint8')
#W = np.random.rand(D, C)
#(loss1, dw1) = softmax_loss_naive(W, X, y, .1)
##print "loss =", loss1
#print "dW = ", dw1
#(loss2, dw2) = softmax_loss_vectorized(W, X, y, .1)
##print "loss =", loss2
#print "dW = ", dw2
