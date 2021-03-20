import matplotlib.pyplot as plt

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

def instatantaneous_error(y,pred):
  """
    Input:
        y: Actual value
        pred: Predicted value
    Output: 
        error: mean square error
      This function calculates squared error for a single data point
  """
  k = len(y)
  error = 0
  for i in range(k):
    error = error + pow(y[i]-pred[i],2)
  return error/2

def avg_error(y, pred):
  """
    Input:
        y: Actual value predictions
        pred: Predicted values
    Output: 
        error: mean square error
      This function calculates average of mean squared error for all data points.
  """
  # error stores total error
  error = 0
  n = len(y)
  for i in range(0,n,1):
    error = error + instatantaneous_error(y[i], pred[i])
  error = error/(2*n)
  return error

def y_single(y_three):
  """
    Input:
        y_three: class vector with column for each class
    Output:
        y_single: Class in single column vector
  """
  y_single = []
  for i in range(len(y_three)):
    for j in range(len(y_three[0])):
      if(y_three[i][j]==1):
        y_single.append(j)
  return y_single
      