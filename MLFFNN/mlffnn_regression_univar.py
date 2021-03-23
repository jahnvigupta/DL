from split_dataset import split_dataset
from MLFFNN_linear_activation import mlffnn_1hidden_layer
from neuron_output_plot import plot_output_1hidden
from reg_value_pred import reg_value_prediction_1hidden
from MLFFNN_utilities import avg_error
from neuron_output_uni import plot_neurons_reg_uni
from decision_plot_MLFFNN import decision_plot_1hidden_MLFFNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read(file_name):
  """
    Input:
        file_name: file to be converted into dataframe
        
    Output: 
        df: file stored as a dataframe
  """
  df = pd.read_csv(file_name, header=None, sep=",", names=[1,2])
  return df

#Reading data into df
df = read("../Group09/Regression/UnivariateData/9.csv")
df[0] = 1
# dividing data into X and y
X = df[[0,1]]
y = df[[2]]
# converting y to numpy array
y = np.array(y)

# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val
X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X,y)

# Plotting Datasets
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset Visualisation")
plt.scatter(X[1],y)
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data Visualisation")
plt.scatter(X_train[1],y_train)
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test Data Visualisation")
plt.scatter(X_test[1],y_test)
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Validation Data Visualisation")
plt.scatter(X_val[1],y_val)

# defining epochs, learning_rate, beta, error_diff
epochs = 500
learning_rate = 0.001
beta = 1
error_diff = 0.000001
J = 1

# Training MLFFNN with 1 hidden layer
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(X_train, y_train, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = reg_value_prediction_1hidden(X_val, wij, wjk, beta)
# computing error for validation data
val_error = avg_error(y_val, predictions_val)
print("Mean squared error in predictions for Validation data: ", val_error)

# Plotting epoch errors for perceptron trained
plt.figure()
plt.scatter(epoch_num, epoch_error)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")

# predictions for test data
predictions_test = reg_value_prediction_1hidden(X_test, wij, wjk, beta)
# computing error for test data
test_error = avg_error(y_test, predictions_test)
print("Mean squared error in predictions for test data: ", test_error)

# predictions for train data
predictions_train = reg_value_prediction_1hidden(X_train, wij, wjk, beta)
# computing error for train data
train_error = avg_error(y_train, predictions_train)
print("Mean squared error in predictions for train data: ", train_error)

# Plotting epoch error for the MLFFNN with one hidden layer
plt.figure()
plt.title("")
plt.scatter(epoch_num, epoch_error)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")

# Plotting Histogram for Training, Test, Validation data mean squared error
plt.bar(['Training_data','Test_data','Validation_data'],[train_error,test_error,val_error])
plt.ylabel("Error")
plt.xlabel("Type of data")
plt.title("Errors in different type of data")

# Plotting predicted and target output for training, test and validation
plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for training data")
plt.scatter(y_train,predictions_train)

plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for test data")
plt.scatter(y_test,predictions_test)

plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for validation data")
plt.scatter(y_val,predictions_val)

# Plotting target output and predicted output with input data for training, test and validation
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training data")
plt.scatter(X_train[1], y_train, c='b')
plt.scatter(X_train[1], predictions_train, c='r')

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test data")
plt.scatter(X_test[1], y_test, c='b')
plt.scatter(X_test[1], predictions_test, c='r')

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training data")
plt.scatter(X_train[1], y_train, c='b')
plt.scatter(X_train[1], predictions_train, c='r')
plt.ylabel("y")
plt.title("Validation data")
plt.scatter(X_val[1], y_val, c='b')
plt.scatter(X_val[1], predictions_val, c='r')

# plotting output for each neuron
plot_neurons_reg_uni(X, wij,wjk, beta)