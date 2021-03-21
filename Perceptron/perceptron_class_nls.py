#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from split_dataset import split_dataset
from perceptron_3class_utilities import replace
from perceptron_3class_utilities import plot_data
from perceptron_3class_utilities import prediction_3
from perceptron_sigmoid import perceptron
from eval_param import confusion_matrix
from eval_param import accuracy
from decision_plot import decision_plot
from decision_plot import decision_plot3

def read(file_name):
  """
    Input:
        file_name: file to be converted into dataframe
        
    Output: 
        df: file stored as a dataframe
  """
  df = pd.read_csv(file_name,  skiprows=1 ,header=None, sep=" ", names=[1, 2], index_col=False)
  df = df[[1, 2]]
  return df

#Reading data into df
df = read("../Group09/Classification/NLS_Group09.txt")

# append column with values 1 for bias term
df[0] = 1
# reordering columns of dataframe
df = df[[0, 1, 2]]

#Reading class1 data into df1
df1 = df.iloc[0:500]
#Reading class2 data into df2
df2 = df.iloc[500:1000]
#Reading class3 data into df3
df3 = df.iloc[1000:1700]

# Plot data before classification
plot_data(df1, df2, df3, "Dataset Visualisation")

# Number of data points for each class is given
#Storing true values for class1 in y1
y1 = 0*np.ones(500)
#Storing true values for class2 in y2
y2 = 1*np.ones(500)
#Storing true values for class3 in y3
y3 = 2*np.ones(700)

# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class1
X_train1, X_test1, X_val1, y_train1, y_test1, y_val1 = split_dataset(df1, y1)
# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class2
X_train2, X_test2, X_val2, y_train2, y_test2, y_val2 = split_dataset(df2, y2)
# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class3
X_train3, X_test3, X_val3, y_train3, y_test3, y_val3 = split_dataset(df3, y3)

plot_data(X_train1, X_train2, X_train3, "Training data visualisation")
plot_data(X_test1, X_test2, X_test3, "Test data visualisation")
plot_data(X_val1, X_val2, X_val3, "Validation data visualisation")

# Concatenating 3 dataframes
X = pd.concat([df1, df2, df3])
y = np.concatenate((y1, y2, y3))
# Getting minimum and maximum values of both features
min_0 = int(min(X[1])-1)
max_0 = int(max(X[1])+1)
min_1 = int(min(X[2])-1)
max_1 = int(max(X[2])+1)


# Concatenating all the training data
X_train = pd.concat([X_train1, X_train2, X_train3])
y_train = np.concatenate((y_train1, y_train2, y_train3))

# Concatenating all the validation data
X_val = pd.concat([X_val1, X_val2, X_val3])
y_val = np.concatenate((y_val1, y_val2, y_val3))

# Concatenating all the test data
X_test = pd.concat([X_test1, X_test2, X_test3])
y_test = np.concatenate((y_test1, y_test2, y_test3))

# defining epochs, learning_rate, beta, error_diff
epochs = 500
learning_rate = 0.001
beta = 1
error_diff = 0.000001

# For first perceptron
# Concatenating first and second training dataframes
X_train12 = pd.concat([X_train1, X_train2])
y_train12 = np.concatenate((y_train1, y_train2))

# Replacing class values to 0 and 1 for training
y_train12_temp = replace(y_train12)

# Training perceptron model 
w12, epoch_num12, epoch_error12 = perceptron(X_train12, y_train12_temp, epochs, learning_rate, beta, error_diff)

# decision plot for first perceptron with training data
decision_plot(w12, beta, min_0, max_0, min_1, max_1)
plt.scatter(df1[1], df1[2], c='r')
plt.scatter(df2[1], df2[2], c='b')
plt.scatter(df3[1], df3[2], c='g')

# for second perceptron
# Concatenating first and second training dataframes
X_train23 = pd.concat([X_train2, X_train3])
y_train23 = np.concatenate((y_train2, y_train3))

# Replacing class values to 0 and 1 for training
y_train23_temp = replace(y_train23)

# Training perceptron model 
w23, epoch_num23, epoch_error23 = perceptron(X_train23, y_train23_temp, epochs, learning_rate, beta, error_diff)

# decision plot for second perceptron with training data
decision_plot(w23, beta, min_0, max_0, min_1, max_1)
plt.scatter(df1[1], df1[2], c='r')
plt.scatter(df2[1], df2[2], c='b')
plt.scatter(df3[1], df3[2], c='g')

# For third perceptron
# Concatenating first and second training dataframes
X_train13 = pd.concat([X_train1, X_train3])
y_train13 = np.concatenate((y_train1, y_train3))

# Replacing class values to 0 and 1 for training
y_train13_temp = replace(y_train13)

# Training perceptron model 
w13, epoch_num13, epoch_error13 = perceptron(X_train13, y_train13_temp, epochs, learning_rate, beta, error_diff)

# decision plot for third perceptron with training data
decision_plot(w13, beta, min_0, max_0, min_1, max_1)
plt.scatter(df1[1], df1[2], c='r')
plt.scatter(df2[1], df2[2], c='b')
plt.scatter(df3[1], df3[2], c='g')

# decision plot for 3 perceptrons with training data
decision_plot3(w12, w23, w13, beta, min_0, max_0, min_1, max_1)
plt.scatter(df1[1], df1[2], c='r')
plt.scatter(df2[1], df2[2], c='b')
plt.scatter(df3[1], df3[2], c='g')

# Plotting epoch errors for 3 perceptron trained
plt.figure()
plt.title("First Perceptron")
plt.scatter(epoch_num12, epoch_error12)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")
plt.figure()
plt.title("Second Perceptron")
plt.scatter(epoch_num23, epoch_error23)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")
plt.figure()
plt.title("Third Perceptron")
plt.scatter(epoch_num13, epoch_error13)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")

# predictions for training data
predictions_train = prediction_3(X_train, w12, w23, w13, beta)
# Confusion matrix for training data
conf_mat_train = confusion_matrix(predictions_train, y_train)
# accuracy for training data
accuracy_train = accuracy(conf_mat_train)

print("Confusion matrix for training data: \n", conf_mat_train)
print("Accuracy for training data: ", accuracy_train)

# predictions for test data
predictions_test = prediction_3(X_test, w12, w23, w13, beta)
# Confusion matrix for test data
conf_mat_test = confusion_matrix(predictions_test, y_test)
# accuracy for test data
accuracy_test = accuracy(conf_mat_test)

print("Confusion matrix for test data: \n", conf_mat_test)
print("Accuracy for test data: ", accuracy_test)

# predictions for validation data
predictions_val = prediction_3(X_val, w12, w23, w13, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Confusion matrix for validation data: \n", conf_mat_val)
print("Accuracy  validation data: ", accuracy_val)