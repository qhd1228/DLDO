# a nearly finished version
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
  return A_G # weighted adjacency matrix of G n by n positive/-1

# original Christofides CO
define SpanningTree_min(A_G)
  #...
  return T_min # boolean adjacency matrix n by n

# our framework NN
def SpanningTree(A_G):
  #...
  return T # boolean adjacency matrix n by n

define OddSet(T)
  #...
  return S # Queue _ by 1

# CO
define Matching(S)
  #...
  return M # regular (0-1) adjacency matrix n by n

define EulerianPath(T,M)
  #...
  return P_E # Queue

define HalmiltonianPath(P_E)
  #...
  return P_H # Queue

define Length(P)
  #...
  return L # scalar

# main function
define Christofides(T)
  #....
  return L


# G_set: N by n by n
# L_set: N by 1
def GenerateTrainingData(G_set):
  # G_set: N * n * n G:adjacency matrix
  # for each G in G_set:
  #   L = algo(G) L : solution of TSP
  #   # get L_set: N * 1
  N = G_set.shape[0]
  L_set = np.zeros(N)
  for i in range(N):
    L_set[i] = algo(G_set[i,:,:])
  return L_set

# solve tsp by Dynamic Programming
def algo(dist):
  N=dist.shape[0]
  path = np.ones((2**(N+1),N))
  dp = np.ones((2**(N+1),N))*-1
 
  def TSP(s,init,num):
      if dp[s][init] !=-1 :
          return dp[s][init]
      if s==(1<<(N)):
          return dist[0][init]
      sumpath=1000000000
      for i in range(N):
          if s&(1<<i):
              m=TSP(s&(~(1<<i)),i,num+1)+dist[i][init]
              if m<sumpath:
                  sumpath=m
                  path[s][init]=i
      dp[s][init]=sumpath
      return dp[s][init]
 

  init_point=0
  s=0
  for i in range(1,N+1):
    s=s|(1<<i)
  distance=TSP(s,init_point,0)
  return distance # return distance

# qiangang
def Vectorize(G):
  # G: n*n
  # upper triangle part, except diagnol
  # V: vector: n*(n-1)/2
  n = len(G)
  #G = G.tolist()
  V = []
  for i in range(n-1):
    V += list(G[i][i+1:])
  return V


def Devectorize(V):
  l = len(V)
  n = 0
  while True:
    if (n-1) * n /2 < l:
      n += 1
    else:
      break
  G = []
  b = 0
  for i in range(n):
    G.append([0 for j in range(i+1)] + V[b:b+ (n-1-i)])
    b += n-1-i
  G = transpose(G)
  return G

def transpose(Gin):
    n = len(Gin)
    Gout = [[0 for ii in range(n)] for jj in range(n)]
#     print(Gout)
    for j in range(n):
        for i in range(n):
            Gout[i][j] = Gin[j][i]
    for j in range(n):
        for i in range(n):
            Gout[i][j] += Gin[i][j]
    return Gout

def TreeGenerate(A):
  # A: the prob matrix a_{ij}\in [0,1]
  B =  1-A
  T = SpanningTree_min(B)
  return T



# training hongda
def training(G_set,L_set):
  # X_train, Y_train
  # vectorize G to X_set
  N = G_set.shape[0]
  n = G_set.shape[1]
  D_x = n*(n-1)/2
  D_y = n*(n-1)/2
  X_train = np.zeros((N,D_x))
  Y_train = L_set
  for i in range(N):
    X_train[i,:] = Vectorize(G_set[i,:,:])
  # paramters
  yita = 0.1
  width = 24
  epoch = 10000
  batch_size = 1
  
 
  w1 = width
  w2 = width
  w3 = D_y
  # Given G with n vertices, we train the NN to obtain T, with cost function J(T) = Chris(T) 
  
  #X=tf.placeholder(tf.float32,[None,n,n]) # G_i maybe need vectorize
  X=tf.placeholder(tf.float32,[None,D_x]) # vectorized G_i=SpanningTree(G_i)
  Y=tf.placeholder(tf.float32,[None,D_y]) # vectorized T
  ## optional
  ## X: G
  ## Y: T
  ## J: d(T,T_data)

  # build the linear NN as l1,l2,l3,l4,.....
  # l1 X*W1+b1
  W_1 = tf.Variable(tf.random.normal([D_x,w1],mean=0, stddev=1.0),name = "W_1")
  b_1 = tf.Variable(tf.random.normal([w1,1],mean=0, stddev=1.0),name = "b_1")

  W_2 = tf.Variable(tf.random.normal([w1,w2],mean=0, stddev=1.0),name = "W_2")
  b_2 = tf.Variable(tf.random.normal([w2,1],mean=0, stddev=1.0),name = "b_2")

  W_3 = tf.Variable(tf.random.normal([w2,w3],mean=0, stddev=1.0),name = "W_3")
  b_3 = tf.Variable(tf.random.normal([w3,1],mean=0, stddev=1.0),name = "b_3")
  # and so on....
  # layer outputs
  l_1 = tf.nn.relu(tf.matmul(tf.transpose(W_1),tf.transpose(X))+b_1) # w1 none
  l_2 = tf.nn.relu(tf.matmul(tf.transpose(W_2),l_1) + b_2) # w2 none
  l_3 = tf.nn.sigmoid(tf.matmul(tf.transpose(W_3),l_2) + b_3) # w3 none || C none

  # the feedback of Neural Network 
  tempLquita =  DeVectorize(l_3) # devectorize
  tempL = DeVectorize(Y)
  U = Christofides(tempLquita)
  V = Christofides(tempL)
  # main objective
  J = 1/2 * tf.square(tf.norm(U-V))

  # train step
  optimizer = tf.train.AdamOptimizer(yita)

  # log the objective
  train = optimizer.minimize(J)

  # init
  init = tf.global_variables_initializer()
  
  # train....
  #....
  value = np.zeros(epoch)
  with tf.Session() as session:.....
    #ini
    session.run(init)
    sum = 0
    # train
    for ep in range(epoch):
      # single epoch
      X_set  = np.zeros((batch_size,D_x))
      Y_set  = np.zeros((batch_size,D_y))
      for index in range(batch_size):   
        # data 
        picker = ep  
        X_set[index,:] = X_train[picker,:]
        Y_set[index,:] = Y_train[picker,:]
      session.run(train, feed_dict = {x:X_set,y:Y_set})
      sum = sum + session.run(J,feed_dict = {x:X_set,y:Y_set})
      value[ep] = sum/ep
    # test
    predicts = session.run(l_3,feed_dict = {x:X_train})
    Lpre = np.zeros(N)
    for i in range(N):
      T = DeVectorize(predicts[:,i])
      Lpre[i] = Christofides(T)
      OpRate = (Lpre[i]-L_set[i])/L_set[i]

    # Plot
    t = range(epoch)
    plt.plot(t,value,label = "1")
    plt.xlabel('epoch')
    plt.ylabel('f')
    plt.legend()
    plt.show()

  # output...
  
