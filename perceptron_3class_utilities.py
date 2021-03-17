import numpy as np
import matplotlib.pyplot as plt
from perceptron_sigmoid import pred

def replace(y):
  """
    Input:
        y: class values

    Output: 
        y: Updated class values
    This function changes the class values of y to to 0 and 1
  """
  a = min(y)
  return np.where(y==a, 0, 1)

def plot_data(df1, df2, df3, title):
  """
    Input:
        df1: dataframe of first class
        df2: dataframe of second class
        df3: dataframe of third class
        title: title of the plot
      This function plots the data points of all the 3 classes
  """
  plt.figure()
  plt.title(title)
  plt.scatter(df1[1], df1[2], c='r')
  plt.scatter(df2[1], df2[2], c='b')
  plt.scatter(df3[1], df3[2], c='g')
  plt.xlabel("Feature1")
  plt.ylabel("Feature2")

def prediction_3(X, w1, w2, w3, beta):
  """
    Input:
        X: Data point for which predictions are to be made
        w1: weight vector of perceptron1
        w2: weight vector of perceptron2
        w3: weight vector of perceptron3
        beta: Parameter for sigmoid function
    Output: 
        predictions: array of predicted class out of all 3 classes for all data points
  """
  # array to store prediction made
  predictions = []
  # iterate through all data points
  for i in range(len(X.index)):
    # Predictions of first perceptron
    pred1 = pred(X.iloc[i], w1, beta)
    if(pred1<=0.5):
      pred1 = 0
    else:
      pred1 = 1
    # Predictions of second perceptron
    pred2 = pred(X.iloc[i], w2, beta)
    if(pred2<=0.5):
      pred2 = 1
    else:
      pred2 = 2
    # Predictions of third perceptron
    pred3 = pred(X.iloc[i], w3, beta)
    if(pred3<=0.5):
      pred3 = 0
    else:
      pred3 = 2
    # Choosing the majority class as prediction
    if(pred1==0 and pred3==0):
      predictions.append(0)
    elif(pred1==1 and pred2==1):
      predictions.append(1)
    elif(pred2==2 and pred3==2):
      predictions.append(2)
    else:
      predictions.append(0)
  return predictions