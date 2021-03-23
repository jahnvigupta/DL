from split_dataset import split_dataset
from MLFFNN_linear_activation import mlffnn_1hidden_layer
from neuron_output_plot import plot_output_1hidden
from reg_value_pred import reg_value_prediction_1hidden
from MLFFNN_utilities import avg_error
from neuron_output_plot_reg import plot_neurons_reg
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
  df = pd.read_csv(file_name, header=None, sep=",", names=[1,2,3])
  return df

#Reading data into df
df = read("../Group09/Regression/BivariateData/9.csv")

# Plotting 3d plot for data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df[[1]], df[[2]], df[[3]])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')
ax.set_title("Complete Data Visualisation")

# adding bias column
df[0] = 1
# dividing data into X and y
X = df[[0,1,2]]
y = df[[3]]
y = np.array(y).flatten()

# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val
X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X,y)
# converting X_train to a dataframe
X_train = pd.DataFrame(X_train)
# convering y_train to a column vector
y_train = (y_train).reshape((len(y_train),1))
# convering y_test to a column vector
y_test = (y_test).reshape((len(y_test),1))
# convering y_val to a column vector
y_val = (y_val).reshape((len(y_val),1))

# Plotting 3d plot for training, test, validation data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train[[1]], X_train[[2]], y_train)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')
ax.set_title("Training Data Visualisation")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[[1]], X_test[[2]], y_test)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')
ax.set_title("Test Data Visualisation")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_val[[1]], X_val[[2]], y_val)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target Variable')
ax.set_title("Validation Data Visualisation")

# defining epochs, learning_rate, beta, error_diff
epochs = 500
learning_rate = 0.001
beta = 1
error_diff = 0.000001
J = 3

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

# Plotting 3d plot for training, test, validation data with target and predicted outputs
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train[[1]], X_train[[2]], y_train, c= 'b', label="Target Values")
ax.scatter3D(X_train[[1]], X_train[[2]], predictions_train, c='r', label="Predicted Values")
ax.legend()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')
ax.set_title("Training Data Visualisation")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[[1]], X_test[[2]], y_test, c= 'b', label="Target Values")
ax.scatter3D(X_test[[1]], X_test[[2]], predictions_test, c='r', label="Predicted Values")
ax.legend()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')
ax.set_title("Test Data Visualisation")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_val[[1]], X_val[[2]], y_val, c= 'b', label="Target Values")
ax.scatter3D(X_val[[1]], X_val[[2]], predictions_val, c='r', label="Predicted Values")
ax.legend()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Output')
ax.set_title("Validation Data Visualisation")

# plotting output for each neuron
plot_neurons_reg(X, wij,wjk, beta)