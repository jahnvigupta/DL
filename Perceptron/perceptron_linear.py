import numpy as np
from sklearn.utils import shuffle

def initialise_w(dim):
  """
    Input:
        dim: dimension of input vector
        
    Output: 
        w: random weight vector
    This function initialises the weights of the perceptron model randomly    
  """
  w = np.random.randn(dim)
  return w

def mean_squared_error(y, s):
  """
    Input:
        y: Actual value 
        s: Predicted value
    Output: 
        error: square error
      This function calculates squared error for a single data point
  """
  error = 0.5*(pow(y-s,2))
  return error

def error(y, pred):
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
    error = error + mean_squared_error(y[i], pred[i])
  error = error/(2*n)
  return error

def pred_linear_act(x, w):
  """
    Input:
        x: Data point for making prediction
        w: weight vector of perceptron
    Output: 
        an: Predicted value of perceptron with linear activation
  """
  # Convert x into a np array
  x = np.array(x)
  an = np.matmul(w.T, x)
  return an

def update_w(w, train_x, y, learning_rate):
  """
    Input:
        w: weight vector of perceptron
        train_x: data point for which prediction need to be made
        y: actual value at data point
        learning_rate: Learning rate of Gradient Descent
        
    Output: 
        w: Updated weight vector
    This function returns the updated weight vector   
  """
  an = pred_linear_act(train_x, w)
  #calculating derivative of error wrt w
  t1 = (y-an)
  t2 = -1*learning_rate*t1
  update = t2*train_x
  #updating weight vector
  w = w - update
  return w

def perceptron(X_train, y_train, epochs, learning_rate, error_diff):
  """
    Input:
        X_train: Training data points
        y_train: actual values for training points
        epochs: Number of epochs for training
        learning_rate: Learning rate for Gradient Descent
        beta: Parameter for sigmoid function
        error_diff: Threshold on error difference between successive epochs

    Output: 
        w: Weight vector after training
        epoch_num: List containing epoch numbers
        epoch_error: List containing errors on every epoch
  """
  # initialising empty list for epoch numbers and epoch error
  epoch_error = []
  epoch_num = []
  # i is iteration number
  i = 0
  # Initialising weights randomly
  w = initialise_w(len(X_train.columns))
  # Running iteration through all the training points till difference between consecutive errors is above threshold
  while(len(epoch_num)<=2 or ((-epoch_error[i-1]+epoch_error[i-2])>error_diff or epoch_error[i-2]<epoch_error[i-1])):
    # Condition learning_rate on iteration number
    if(i<3):
      learning_rate = 0.1
    elif(i<200):
      learning_rate = 0.001
    else:
      learning_rate = 0.0001
    # Shuffle training data for each epoch
    X_train, y_train = shuffle(X_train, y_train)
    # Updating weight vector for every data point
    for j in range(len(X_train.index)):
      w = update_w(w, X_train.iloc[j], y_train[j], learning_rate)
    # predictions is used for storing the predictions of the model for training data
    predictions = np.zeros(len(X_train.index))
    for k in range(len(X_train.index)):
      predictions[k] = pred_linear_act(X_train.iloc[k], w)
    # epoch_error stores error for each epoch
    epoch_error.append(error(y_train, predictions))
    # epoch_num stores the epoch number
    epoch_num.append(i)
    i = i+1
  return w, epoch_num, epoch_error