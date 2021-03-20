import numpy as np
import math

def sigmoid(x, beta):
  """
    Input:
        x: Input value for sigmoid function
        beta: Parameter for sigmoid function
    Output: 
        val: Output value of sigmoid function
  """
  val = 1/(1+math.exp(-1*beta*x))
  return val

def sigmoid_vector(v, beta):
  """
    Input:
        v: Vector for which sigmoid function need to be found
        beta: Parameter for sigmoid function
    Output: 
        sig_val: output vector with sigmoid function applied on input vector
  """
  k = len(v)
  sig_val = []
  for i in range(k):
    sig_val.append(sigmoid(v[i],beta))
  return np.array(sig_val)