#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#cleaned up vers. of random graph generation; readily converted to tensor format b/c already in np.array format

import numpy as np

# G loading G1,....,GN, L1,...,LN
# Gi,Li come from dataset or by generating
# A_G of G

import random

def random_adjmat(n,max_size):   
    #generates a single instance of adjacency matrix for a Cartesian graph
    coord=np.array([[random.randint(0, max_size) for j in range(2)] for i in range(n)])
    
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
    
result = set_gen(2,3,10)
print(result)

