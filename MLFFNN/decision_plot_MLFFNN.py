import pandas as pd
import numpy as np
from class_predic_MLFFNN import class_prediction_1hidden
import matplotlib.pyplot as plt
from class_predic_MLFFNN import class_prediction_2hidden

def decision_plot_1hidden_MLFFNN(wij, wjk, beta, x1, x2, y1, y2):
  """
    Input:
        wij: Weight vector from input to hidden layer after training
        wjk: Weight vector from hidden to output layer after training
        beta: parameter for sigmoid function
        x1: min value of xrange
        x2: max value of xrange
        y1: min value of yrange
        y2: max value of yrange
    This function plots the decision plot of given MLFFNN with 1 hidden layer
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
  predictions = class_prediction_1hidden(df, wij, wjk, beta)

  # Storing index for classes
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

def decision_plot_2hidden_MLFFNN(wij1, wj1j2, wj2k, beta, x1, x2, y1, y2):
  """
    Input:
        wij1: weight vector from input layer to 1st hidden layer
        wj1j2: weight vector from 1st hidden layer to 2nd hidden layer
        wj1j2: weight vector from 2nd hidden layer to output layer
        x1: min value of xrange
        x2: max value of xrange
        y1: min value of yrange
        y2: max value of yrange
    This function plots the decision plot of given MLFFNN with 1 hidden layer
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
  predictions = class_prediction_2hidden(df, wij1, wj1j2, wj2k, beta)

  # Storing index for classes
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