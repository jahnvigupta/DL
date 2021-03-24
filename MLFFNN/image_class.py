from sklearn.model_selection import train_test_split
from class_predic_MLFFNN import class_prediction_1hidden
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eval_param import confusion_matrix
from eval_param import accuracy
import numpy as np
from sigmoid import sigmoid_vector
from MLFFNN_utilities import avg_error
from sklearn.utils import shuffle

def initialise_weights(J, K):
  """
    Input:
        J: Number of neurons in first layer
        K: Number of neurons in second layer
        
    Output: 
        w: random weight vector of dimension (J+1)*K
    This function initialises the weights of the Neural Network from first layer to second layer 
  """
  w = np.random.rand(J,K)
  return w

def feedforward_1hidden(x,wij,wjk,beta):
  """
    Input:
        x: Vector for which feedforward computation is to be done
        wij: weight matrix from input to hidden layer
        wjk: weight matrix from hidden to output layer
        beta: Parameter for sigmoid function
    Output: 
        anj: activation value from hidden layer
        snj: output from hidden layer
        ank: activation value from output layer
        snk: output from output layer
  """
  # Computation through hidden layer
  anj = wij.T.dot(x)
  snj = sigmoid_vector(anj, beta)
  snj_bias = np.append(1,snj) # snj with bias term

  # computation through output layer
  ank = wjk.T.dot(snj_bias)
  snk = sigmoid_vector(ank, beta)
  return anj, snj, ank, snk

def update_wjk_1hidden(yk, sk, ak, beta, sj_bias, learning_rate):
  """
    Input:
        yk: Actual prediction
        sk: Prediction made by network
        ak: activation value from output layer
        beta: parameter of sigmoid function
        sj_bias: input to output layer
        learning_rate: Learning Rate of Gradient Descent
        
    Output: 
        update: Update to wjk
  """
  # derivative of f(ak)
  d_f = beta*sk*(1-sk)
  delta_k = (yk-sk)*d_f
  update = learning_rate*np.outer(sj_bias,delta_k)
  return update

def update_wij_1hidden(x, yk, sk, ak, wjk, beta, sj, learning_rate):
  """
    Input:
        x: Input to Neural Network
        yk: Actual prediction
        sk: Prediction made by network
        ak: activation value through output layer
        wjk: Weight matrix from hidden to output layer
        beta: parameter of sigmoid function
        sj: output from hidden layer
        learning_rate: Learning Rate of Gradient Descent
        
    Output: 
        update: Update to wjk
  """
  d_fk = beta*sk*(1-sk)
  delta_k = (yk-sk)*d_fk
  temp = np.dot(delta_k,wjk[1:,:].T)
  d_fj = beta*sj*(1-sj)
  delta_j = temp*d_fj
  update = learning_rate*np.outer(x, delta_j)
  return learning_rate*update

def mlffnn_1hidden_layer(X_train, y_train, learning_rate, beta, J, error_thresh):
  """
    Input:
        X_train: Training data points
        y_train: actual values for training points
        learning_rate: Learning rate for Gradient Descent
        beta: Parameter for sigmoid function
        J: Number of neurons in hidden layer
    Output: 
        wij: Weight vector from input to hidden layer after training
        wjk: Weight vector from hidden to output layer after training
        epoch_num: List containing epoch numbers
        epoch_error: List containing errors on every epoch
  """
  # initialising empty list for epoch numbers and epoch error
  epoch_error = []
  epoch_num = []
  # i is iteration number
  i = 0

  # Initialising weights from input layer to hidden layer randomly
  wij = initialise_weights(len(X_train.columns),J)
  # Initialising weights from hidden layer output layer randomly
  wjk = initialise_weights(J+1,len(y_train[0]))
  
  # Converting X_train to numpy array
  X_train = np.array(X_train)
  # Running iteration through all the training points till difference between consecutive errors is above threshold
  while(len(epoch_num)<=2 or (epoch_error[i-1])>error_thresh):
    X_train, y_train = shuffle(X_train, y_train)
    # predictions is used for storing the predictions of the model for training data
    predictions = []
    # Updating weight vector for every data point
    for j in range(len(X_train)):
      # Feedforward part
      anj, snj, ank, snk = feedforward_1hidden(X_train[j],wij,wjk,beta)
      snj_bias = np.append(1,snj)
      predictions.append(snk)

      # backpropogation
      wjk_copy = wjk
      wjk = wjk + update_wjk_1hidden(y_train[j], snk, ank, beta, snj_bias, learning_rate)
      wij = wij + update_wij_1hidden(X_train[j], y_train[j], snk, ank, wjk_copy, beta, snj, learning_rate)

    # epoch_num stores the epoch number
    epoch_num.append(i)
    # epoch_error stores error for each epoch
    epoch_error.append(avg_error(y_train, predictions))
    i = i+1
  return epoch_error, epoch_num, wij, wjk

# loading data from text files
bovw_train = np.loadtxt("image_train.txt").reshape(150, 33)
bovw_test = np.loadtxt("image_test.txt").reshape(150, 33)

# extracting input and output features in separate arrays
X = bovw_train[:,0:32]
y  = bovw_train[:,32]
X_test = bovw_test[:,0:32]
y_test  = bovw_test[:,32]

# Dividing the data into different classes
# class 1
X1 = X[0:50,:]
y1 = y[0:50]

# class 2
X2 = X[50:100,:]
y2 = y[50:100]

# class3
X3 = X[100:150,:]
y3 = y[100:150]

# splitting given data of different classes into train and validation data
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y1, test_size=0.2, random_state=1)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y2, test_size=0.2, random_state=1)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X3, y3, test_size=0.2, random_state=1)

# concatenating all the training data
X_train = np.concatenate([X_train1,X_train2,X_train3])
y_train = np.concatenate([y_train1,y_train2,y_train3])

# concatenating all the validation data
X_val = np.concatenate([X_val1,X_val2,X_val3])
y_val = np.concatenate([y_val1,y_val2,y_val3])

# converting single column output vectors to 3 column output vectors
y_train_col = y_3columns(y_train)
y_test_col = y_3columns(y_test)
y_val_col = y_3columns(y_val)

# defining learning_rate, beta
learning_rate = 0.1
beta = 1

error_thresh = 0.05
# Training MLFFNN with 1 hidden layer
J = 32
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(pd.DataFrame(X_train), y_train_col, learning_rate, beta, J, error_thresh)

print("Cross Validation with MLFFNN with 1hidden layers with 32 neurons")
# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Confusion Matrix for training data : \n", conf_mat_val)
print("Accuracy for validation data is : ", accuracy_val)
print("\n")

# predictions for test data
predictions_test = class_prediction_1hidden(X_test, wij, wjk, beta)
# Confusion matrix for test data
conf_mat_test = confusion_matrix(predictions_test, y_test)
# accuracy for test data
accuracy_test = accuracy(conf_mat_test)

print("Accuracy for test data is : ", accuracy_test)
print("Confusion Matrix for testing data : \n", conf_mat_test)
print("\n")

# predictions for training data
predictions_train = class_prediction_1hidden(X_train, wij, wjk, beta)
# Confusion matrix for training data
conf_mat_train = confusion_matrix(predictions_train, y_train)
# accuracy for training data
accuracy_train = accuracy(conf_mat_train)

print("Accuracy for training data is : ", accuracy_train)
print("Confusion Matrix for training data : \n", conf_mat_train)
print("\n")

# Plotting epoch error for the MLFFNN with one hidden layer
plt.figure()
plt.title("")
plt.scatter(epoch_num, epoch_error)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")