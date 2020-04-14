import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import seed, randint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# G loading G1,....,GN, L1,...,LN
# Gi,Li come from dataset or by generating
# A_G of G
define AG(G)
  #....
  return A_G # weighted adjacency matrix of G n by n

# original Christofides CO
define SpanningTree_min(A_G)
  #...
  return T_min # weighted adjacency matrix n by n

# our framework NN
define SpanningTree(A_G)
  #...
  return T # weighted adjacency matrix n by n

define OddSet(T)
  #...
  return S # Boolean vector 0-1 n by 1 or Queue _ by 1 *?

# CO
define Matching(S)
  #...
  return M

define EulerianPath(T,M)
  #...
  return P_E

define HalmiltonianPath(P_E)
  #...
  return P_H

# main function
define Christofides(T)
  #....
  return L


# training
define training(G_set,L_set)
  # l1,l2,l3,l4,.....
  # X=tf.placeholder(tf.float32,[None,n,n]) # G_i maybe need vectorize
  X=tf.placeholder(tf.float32,[None,n,n]) # T_i=SpanningTree(G_i)
  Y=tf.placeholder(tf.float32,[None,1]) # L
  # l1
  W1=...
  B1=...
  # l2
  W2=...
  B2=...
  # the feedback of Neural Network 
  J = Christofides(X)
  f = tf.square(tf.norm(J-Y))

  # train step
  optimizer = tf.train.AdamOptimizer(yita)

  # log the objective
  train = optimizer.minimize(f)

  # init
  init = tf.global_variables_initializer()

  # train....
  #....

  #...
  #Plot.....
  #output...