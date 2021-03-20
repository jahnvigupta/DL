import numpy as np
from MLFFNN import feedforward_2hidden
from MLFFNN import feedforward_1hidden

def class_prediction_1hidden(X, wij, wjk, beta):
  """
    Input:
        X: data for which classes are to be predicted
        wij: weight vector from input layer to hidden layer
        wjk: weight vector from hidden layer to output layer
        beta: parameter of sigmoid function
    Output:
        predictions: prediction of class assigned to data points
    This function returns the class based on values of sigmoid function of MLFFNN with 1 hidden layer
  """
  predictions = []
  # converting X to array
  X = np.array(X)
  for i in range(len(X)):
    # computing output through MLFFNN
    anj, snj, ank, snk = feedforward_1hidden(X[i],wij,wjk,beta)
    # assigning class based on output from sigmoid activation layer
    if(snk[0]>snk[1] and snk[0]>snk[2]):
      predictions.append(0)
    elif(snk[1]>snk[0] and snk[1]>snk[2]):
      predictions.append(1)
    elif(snk[2]>snk[1] and snk[2]>snk[0]):
      predictions.append(2)
    else:
      predictions.append(0)
  return predictions

def class_prediction_2hidden(X, wij1, wj1j2, wj2k, beta):
  """
    Input:
        X: data for which classes are to be predicted
        wij1: weight vector from input layer to 1st hidden layer
        wj1j2: weight vector from 1st hidden layer to 2nd hidden layer
        wj1j2: weight vector from 2nd hidden layer to output layer
        beta: parameter of sigmoid function
    Output:
        predictions: prediction of class assigned to data points
    This function returns the class based on values of sigmoid function of MLFFNN with 1 hidden layer
  """
  predictions = []
  # converting X to array
  X = np.array(X)
  for i in range(len(X)):
    # computing output through MLFFNN
    anj1, snj1, anj2, snj2, ank, snk = feedforward_2hidden(X[i],wij1,wj1j2,wj2k,beta)
    # assigning class based on output from sigmoid activation layer
    if(snk[0]>snk[1] and snk[0]>snk[2]):
      predictions.append(0)
    elif(snk[1]>snk[0] and snk[1]>snk[2]):
      predictions.append(1)
    elif(snk[2]>snk[1] and snk[2]>snk[0]):
      predictions.append(2)
    else:
      predictions.append(0)
  return predictions