import numpy as np
from MLFFNN_linear_activation import feedforward_1hidden
import numpy as np

def reg_value_prediction_1hidden(X, wij, wjk, beta):
  """
    Input:
        X: data for which classes are to be predicted
        wij: weight vector from input layer to hidden layer
        wjk: weight vector from hidden layer to output layer
        beta: parameter of sigmoid function
    Output:
        predictions: prediction of regression trained function value
  """
  predictions = []
  # converting X to array
  X = np.array(X)
  for i in range(len(X)):
    # computing output through MLFFNN
    anj, snj, ank, snk = feedforward_1hidden(X[i],wij,wjk,beta)
    predictions.append(np.array(snk))
  return predictions
