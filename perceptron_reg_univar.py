# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from split_dataset import split_dataset
from perceptron_linear import perceptron
from perceptron_linear import error
from perceptron_linear import pred_linear_act

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
df = read("Group09/Regression/UnivariateData/9.csv")
df[0] = 1
# dividing data into X and y
X = df[[0,1]]
y = df[[2]]
y = np.array(y).flatten()

# defining epochs, learning_rate, beta, error_diff
epochs = 500
learning_rate = 0.001
error_diff = 0.000001

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

# Training perceptron model 
w, epoch_num, epoch_error = perceptron(X_train, y_train, epochs, learning_rate, error_diff)

# Plotting epoch errors for perceptron trained
plt.figure()
plt.scatter(epoch_num, epoch_error)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")

# making predictions for training data
pred_train = []
for i in range(len(X_train)):
  pred_train.append(pred_linear_act(X_train.iloc[i], w))

# making predictions for test data
pred_test = []
for i in range(len(X_test)):
  pred_test.append(pred_linear_act(X_test.iloc[i], w))

# making predictions for validation data
pred_val = []
for i in range(len(X_val)):
  pred_val.append(pred_linear_act(X_val.iloc[i], w))

# Plotting target output and predicted output with input data for training, test and validation
plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training data")
plt.scatter(X_train[1], y_train, c='b')
plt.scatter(X_train[1], pred_train, c='r')

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test data")
plt.scatter(X_test[1], y_test, c='b')
plt.scatter(X_test[1], pred_test, c='r')

plt.figure()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training data")
plt.scatter(X_train[1], y_train, c='b')
plt.scatter(X_train[1], pred_train, c='r')
plt.ylabel("y")
plt.title("Validation data")
plt.scatter(X_val[1], y_val, c='b')
plt.scatter(X_val[1], pred_val, c='r')

# calculating error for training, test, validation data
training_error = error(y_train,pred_train)
test_error = error(y_test,pred_test)
val_error = error(y_val,pred_val)
# Plotting Histogram for Training, Test, Validation data mean squared error
plt.figure()
plt.bar(['Training_data','Test_data','Validation_data'],[training_error,test_error,val_error])
plt.ylabel("Error")
plt.xlabel("Type of data")
plt.title("Errors in different type of data")

print("Mean squared error in predictions for Training data: ", training_error)
print("Mean squared error in predictions for Test data: ", test_error)
print("Mean squared error in predictions for Validation data: ", val_error)

# Plotting predicted and target output for training, test and validation
plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for training data")
plt.scatter(y_train,pred_train)

plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for test data")
plt.scatter(y_test,pred_test)

plt.figure()
plt.xlabel("Target Output")
plt.ylabel("Predicted Output")
plt.title("Predicted vs target output for validation data")
plt.scatter(y_val,pred_val)