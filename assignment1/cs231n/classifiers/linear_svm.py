import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for n in xrange(num_train):
    scores = X[n].dot(W)
    correct_class_score = scores[y[n]]
    for c in xrange(num_classes):
      if c == y[n]:
        dW[:, c] -= (np.sum((scores - correct_class_score + 1) > 0) - 1) * X[n]
        continue
      margin = scores[c] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, c] += X[n]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW / num_train + reg * 2 * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  D = W.shape[0]
  C = W.shape[1]
  S = X.dot(W)
  adjusted = S - S[(xrange(N), y)].reshape(-1, 1) + 1
  adjusted[(xrange(N), y)] = 0
  multiplier = (adjusted > 0) * 1
  adjusted = multiplier * adjusted
  loss = np.sum(adjusted) / N + reg * np.sum(W*W)
  multiplier[(xrange(N), y)] = -np.sum(multiplier, axis=1)
  dW = X.T.dot(multiplier)/N + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
#N = 10
#D = 1000
#C = 10
#X = np.random.rand(N, D)
#y = np.random.rand(N) * 10
#y = y.astype('uint8')
#W = np.random.rand(D, C)
#(loss1, dw1) = svm_loss_naive(W, X, y, .1)
#print "loss =", loss1
#print "dW = ", dw1
#(loss2, dw2) = svm_loss_vectorized(W, X, y, .1)
#print "loss =", loss2
#print "dW = ", dw2
#print "diff loss = ", loss2 - loss1
#print "diff dW= ", np.sum(np.abs(dw2 - dw1))
