import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_3class_utilities import prediction_3
from perceptron_sigmoid import pred

def decision_plot(w, beta, x1, x2, y1, y2):
  """
    Input:
        w: weights of perceptron model
        beta: parameter for sigmoid function
        x1: min value of xrange
        x2: max value of xrange
        y1: min value of yrange
        y2: max value of yrange
    This function plots the decision plot of given model of perceptron model
  """
  # define the x and y scale
  x1grid = np.arange(x1, x2, 0.1)
  x2grid = np.arange(y1, y2, 0.1)
  # create all of the lines and rows of the grid
  xx, yy = np.meshgrid(x1grid, x2grid)
  # flatten each grid to a vector
  r1, r2 = xx.flatten(), yy.flatten()
  r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
  # horizontal stack vectors to create x1,x2 input for the model
  grid = np.hstack((r1,r2))

  # Converting grid to dataframe
  df = pd.DataFrame(grid, columns = [1,2])
  df[0] = 1
  df = df[[0,1,2]]
  # Making predictions for the created dataframe
  predictions = []
  for k in range(len(df.index)):
      predictions.append(pred(df.iloc[k], w, beta))

  # Storing index for classes
  index1 = []
  index2 = []
  for i in range(len(predictions)):
    if(predictions[i]<=0.5):
      index1.append(i)
    else:
      index2.append(i)
  
  # Plotting decision plot according to class
  plt.figure()
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.scatter(df.iloc[index1][1],df.iloc[index1][2],c='c')
  plt.scatter(df.iloc[index2][1],df.iloc[index2][2],c='m')

def decision_plot3(w1, w2, w3, beta, x1, x2, y1, y2):
  """
    Input:
        w1: weights of first perceptron
        w2: weights of second perceptron
        w3: weights of third perceptron
        beta: parameter for sigmoid function
        x1: min value of xrange
        x2: max value of xrange
        y1: min value of yrange
        y2: max value of yrange
    This function plots the decision plot of given model of 3 perceptrons
  """
  # define the x and y scale
  x1grid = np.arange(x1, x2, 0.1)
  x2grid = np.arange(y1, y2, 0.1)
  # create all of the lines and rows of the grid
  xx, yy = np.meshgrid(x1grid, x2grid)
  # flatten each grid to a vector
  r1, r2 = xx.flatten(), yy.flatten()
  r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
  # horizontal stack vectors to create x1,x2 input for the model
  grid = np.hstack((r1,r2))

  # Converting grid to dataframe
  df = pd.DataFrame(grid, columns = [1,2])
  df[0] = 1
  df = df[[0,1,2]]
  # Making predictions for the created dataframe
  predictions = prediction_3(df, w1, w2, w3, beta)

  # Storing index for values 0,1, 2
  index1 = []
  index2 = []
  index3 = []
  for i in range(len(predictions)):
    if(predictions[i]==0):
      index1.append(i)
    elif(predictions[i]==1):
      index2.append(i)
    else:
      index3.append(i)
  
  # Plotting decision plot according to class
  plt.figure()
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.scatter(df.iloc[index1][1],df.iloc[index1][2],c='c')
  plt.scatter(df.iloc[index2][1],df.iloc[index2][2],c='m')
  plt.scatter(df.iloc[index3][1],df.iloc[index3][2],c='y')