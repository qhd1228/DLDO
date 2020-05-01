import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import seed, randint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Traditional Christofides ############################################################################################################################
def tsp(graph):
    G = {}
    for this in range(len(graph)):
        for another_point in range(len(graph)):
            if this != another_point:
                if this not in G:
                    G[this] = {}
                G[this][another_point] = graph[this][another_point]
    # build a minimum spanning tree
    MSTree = minimum_spanning_tree_T(G)
    #print("MSTree: ", MSTree)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes_T(MSTree)
    #print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching_T(MSTree, G, odd_vertexes)
    #print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour_T = find_eulerian_tour_T_T(MSTree, G)

    #print("Eulerian tour: ", eulerian_tour_T)

    current = eulerian_tour_T[0]
    path = [current]
    visited = [False] * len(eulerian_tour_T)
    visited[0] = True

    length = 0

    for v in eulerian_tour_T[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    path.append(path[0])

    #print("Result path: ", path)
    #print("Result length of the path: ", length)

    return length

def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)

def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}

                graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1])

    return graph

class UnionFind_T:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

def minimum_spanning_tree_T(G):
    tree = []
    subtrees = UnionFind_T()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append([u, v, W])
            subtrees.union(u, v)
    return tree

def find_odd_vertexes_T(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes

def minimum_weight_matching_T(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        #print(v)
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)

def find_eulerian_tour_T_T(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST_T(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP

def remove_edge_from_matchedMST_T(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST

# Learning Christofides###########################################################################################################################
# G loading G1,....,GN, L1,...,LN
# Gi,Li come from dataset or by generating
# A_G of G
# mst
def Prim(graph):
    vertex_num = graph.shape[0]
    T = np.zeros((vertex_num,vertex_num))
    INF = 1 << 10
    for i in range(vertex_num):
        for j in range(vertex_num):
            if graph[i,j] == 0 and i!=j:
                graph[i,j] = INF
    #print(graph)
    visit = [False] * vertex_num
    dist = [INF] * vertex_num
    aset = np.zeros(vertex_num)
    bset = np.zeros(vertex_num)

    for i in range(vertex_num):

        minDist = INF + 1
        nextIndex = -1

        for j in range(vertex_num):
            if dist[j] < minDist and not visit[j]:
                minDist = dist[j]
                nextIndex = j
                aset[i] = j
        #print (nextIndex)
        visit[nextIndex] = True

        for j in range(vertex_num):
            if dist[j] > graph[nextIndex][j] and not visit[j]:
                dist[j] = graph[nextIndex][j]
                bset[i] = j
                #preIndex[j] = nextIndex
    
    for k in range(vertex_num-1):
        i = k+1
        acor = int(aset[i])
        bcor = int(bset[i])
        T[acor,bcor] = dist[i]
        T[bcor,acor] = dist[i]
               

    return T #preIndex

def DeWeighted(G, T):
    N = len(G)
    checker = np.zeros((3,3))
    if type(G) != type(checker):
      for i in range(N):
        for j in range(N):
          G[i][j] = tf.cast(G[i][j], tf.float32)
          G[i][j] = tf.reshape(G[i][j],[1])

    Tb = G
    Tout = [[-1,-1,1.000]]
    for i in range(N):
        for j in  range(N):
            Tb[i][j] = T[i][j]*G[i][j]
            if type(G) != type(checker):
              judge = tf.cast(Tb[i][j] == 0, tf.bool)
              temp = [[i,j,tf.cast(G[i][j],tf.float32)]]
              re = tf.cond(judge, lambda: None == 0, lambda: Tout.extend(temp) == 0)  
            if type(G) == type(checker):
              if Tb[i][j] != 0:
                temp = [[i,j,G[i][j]]]
                Tout.extend(temp)
            #Tb[i][j] = G[i][j]
    Tout.pop(0)
    return Tout

def Christofides(G, T):
    G_save = np.array(G)
    MSTree = DeWeighted(G,T)
    G = G_save
    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
    #print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
    #print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)

    #print("Eulerian tour: ", eulerian_tour)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)
    visited[0] = True

    length = 0

    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    #path.append(path[0])
    #print('Predicted Path', path)

    return length

def Christofides_debug(G, T, G_debug):

    MSTree = DeWeighted(G,T)
    
    G = G_debug
    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
    #print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
    #print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)

    #print("Eulerian tour: ", eulerian_tour)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)
    visited[0] = True

    length = 0

    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    path.append(path[0])
    print('Predicted Path', path)
    #print("Result path: ", path)
    #print("Result length of the path: ", length)

    return length, path

class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest

def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree

def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes

def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)
    N = len(G)
    checker = np.zeros((3,3))
    if type(G) != type(checker):
      for i in range(N):
        for j in range(N):
          G[i][j] = tf.cast(G[i][j], tf.float32)
          G[i][j] = tf.reshape(G[i][j],[1])
    while odd_vert:
        #print(1)
        v = odd_vert.pop()
        length = int(9999)
        u = 1
        closest = 0
        for u in odd_vert:
            #with tf.Session().as_default():
            #  print('u=', u.eval())
            judge1 = tf.cast(v!=u, tf.bool)
            vint = tf.cast(v, tf.int32)
            uint = tf.cast(u, tf.int32)
            index = tf.Variable([vint[0],uint[0]], dtype = tf.int32)
            G = tf.cast(G,tf.int32)
            gvu = tf.gather_nd(G, index)
            judge2 = tf.cast(gvu[0] < length, tf.bool)
            #if v != u and G[v][u] < length:
            
            length = tf.cond(judge1 & judge2, lambda:gvu, lambda:length)
            closest = tf.cond(judge1 & judge2, lambda:uint[0], lambda:closest)

        MST.append((v, closest, length))
        if closest in set(odd_vert):
          odd_vert.remove(closest)

def find_eulerian_tour(MatchedMSTree, G):
    # find neigbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]
    #print(len(MatchedMSTree))
    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            #print(i)
            if len(neighbours[v]) > 0:
                break
        a = len(neighbours[v])
        #print(len(neighbours[v]))
        v = v
        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)
            #print(len(MatchedMSTree))
            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP

def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST

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

def Vectorize(G, n):
  # G: n*n
  # upper triangle part, except diagnol
  # V: vector: n*(n-1)/2
  #n = len(G)
  #G = G.tolist()
  V = []
  for i in range(n-1):
    V += list(G[i][i+1:])
  return V

def DeVectorize(V, n):
  l =  int(n*(n-1)/2)
  G = [[0 for ii in range(n)] for jj in range(n)]
  b = 0
  for i in range(n):
    for j in range(n-i-1):
      #print(i, j)
      #print(i, i+j+1)
      G[i][j + i + 1] = V[b + j]
      G[j + i + 1][i] = V[b + j]
    b += n-1-i
  return G

def transpose(Gin, n):
    # n = len(Gin)
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
  B =  1-np.array(A)
  for i in range(len(B)):
    B[i,i] = 0
  T = Prim(B)
  return T

def MST_training(graph):
    N = len(graph)
    G = {}
    for this in range(len(graph)):
        for another_point in range(len(graph)):
            if this != another_point:
                if this not in G:
                    G[this] = {}
                G[this][another_point] = graph[this][another_point]
    tree = []
    subtrees = UnionFind_T()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append([u, v, W])
            subtrees.union(u, v)
    A_t = np.zeros((N,N))
    K = len(tree)
    for k in range(K):
        i = tree[k][0]
        j = tree[k][1]
        length = tree[k][2]
        A_t[i][j] = length
    return A_t

# training hongda
def training(G_set,L_set):
  G_set_save = np.array(G_set)
  # X_train, Y_train
  # vectorize G to X_set
  N = int(G_set.shape[0])
  n = int(G_set.shape[1])
  #print(n)
  D_x = int(n*(n-1)/2)
  D_y = int(n*(n-1)/2)

  L_TC = np.zeros(N)
  for i in range(N):
    G = G_set[i,:,:]
    # L_TC[i] = tsp(G)
  X_train = np.zeros((N,D_x)) 
  Y_train = np.zeros((N,D_y)) 
  
  for i in range(N):
    X_train[i,:] = Vectorize(G_set[i,:,:], n)
    Tree = MST_training(G_set[i,:,:])
    Y_train[i,:] = Vectorize(Tree, n)
  # paramters
  yita = 0.05
  width = 64
  epoch = 200
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
  J = 0
  for i in range(batch_size):
    tempG = DeVectorize(tf.reshape(X[i,:],[-1,1]), n)
    tempLquita =  DeVectorize(tf.reshape(l_3[:,i],[-1,1]), n) # devectorize 
    tempL = DeVectorize(tf.reshape(Y[i,:],[-1,1]), n)
    U = Christofides(tempG, tempLquita)
    V = Christofides(tempG, tempL)
    # main objective
    J = J + 1/2 * tf.square(tf.norm(U-V))

  # train step
  optimizer = tf.train.AdamOptimizer(yita)

  # log the objective
  train = optimizer.minimize(J)

  # init
  init = tf.global_variables_initializer()
  
  # train....
  #....
  value = np.zeros(epoch)
  with tf.Session() as session:
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
        picker = np.random.randint(0,N)
        X_set[index,:] = X_train[picker,:]
        Y_set[index,:] = Y_train[picker,:]
      session.run(train, feed_dict = {X:X_set,Y:Y_set})
      tempV = session.run(J,feed_dict = {X:X_set,Y:Y_set})
      sum = sum + tempV
      print(tempV)
      value[ep] = sum/ep
    # test
    predicts = session.run(l_3,feed_dict = {X:X_train})
    # path_pre = np.zeros((N,n+1))
    L_pre = np.zeros(N)
    for i in range(N):
      G = G_set[i,:,:]
      tempT = DeVectorize(predicts[:,i], n)
      T = TreeGenerate(tempT)
      L_pre[i] = Christofides(G,T)
      OpRate = (L_pre[i]-L_set[i])/L_set[i]
      # print('case: ', i, 'DP', L_set[i], 'TC', L_TC[i], 'Learning Christofides', L_pre[i])
      print('case: ', i, 'DP', L_set[i],
       'Learning Christofides', L_pre[i], "OpRate", OpRate)
    # Plot
    t = range(epoch)
    plt.plot(t,value,label = "1")
    plt.xlabel('epoch')
    plt.ylabel('f')
    plt.legend()
    plt.show()


# test part
# Gi,Li come from dataset or by generating
# A_G of G

import random

def random_adjmat(n,max_size):   
    #generates a single instance of adjacency matrix for a Cartesian graph
    coord=np.array([[random.randint(0, max_size) for j in range(2)] for i in range(n)])
    data = []
    for i in range(len(coord)):
      data.append(coord[i,:])
    matrix=np.zeros((n,n))
    # No vertex connects to itself
    
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(i+1,n):
            matrix[i][j] = np.linalg.norm(coord[i,:]-coord[j,:])
            matrix[j][i] = matrix[i][j]

    return matrix

def set_gen(n_case,n,max_size):
    #generates a set of graphs according to random_adjmat stored in three-dimensional NP array format
    cases=np.zeros((n_case,n,n))
    for i in range(n_case):
        cases[i,:,:]=random_adjmat(n,max_size)
    return cases
    
G_set = np.array(set_gen(50,20,20))
L_set = GenerateTrainingData(G_set)

training(G_set, L_set)