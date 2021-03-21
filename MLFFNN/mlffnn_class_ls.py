from split_dataset import split_dataset
from MLFFNN_utilities import plot_data
from MLFFNN_utilities import y_single
from MLFFNN import mlffnn_1hidden_layer
from neuron_output_plot import plot_output_1hidden
from class_predic_MLFFNN import class_prediction_1hidden
from decision_plot_MLFFNN import decision_plot_1hidden_MLFFNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eval_param import confusion_matrix
from eval_param import accuracy

def read(file_name):
  """
    Input:
        file_name: file to be converted into dataframe
        
    Output: 
        df: file stored as a dataframe
  """
  df = pd.read_csv(file_name, header=None, sep=" ", names=[1, 2])
  return df

#Reading class1 data into df1
df1 = read("../Group09/Classification/LS_Group09/Class1.txt")
#Reading class2 data into df2
df2 = read("../Group09/Classification/LS_Group09/Class2.txt")
#Reading class3 data into df3
df3 = read("../Group09/Classification/LS_Group09/Class3.txt")

#append column with values 1 for bias term
df1[0] = 1
df2[0] = 1
df3[0] = 1

#reordering columns of dataframe
df1 = df1[[0, 1, 2]]
df2 = df2[[0, 1, 2]]
df3 = df3[[0, 1, 2]]

# Making True Labels for dataset
# setting first column as 1 for class1 
y1 = np.zeros(((len(df1.index)),3))
y1[:,0] = 1

# setting second column as 1 for class2
y2 = np.zeros(((len(df2.index)),3))
y2[:,1] = 1

# setting third column as 1 for class3 
y3 = np.zeros(((len(df3.index)),3))
y3[:,2] = 1

# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class1
X_train1, X_test1, X_val1, y_train1, y_test1, y_val1 = split_dataset(df1, y1)
# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class2
X_train2, X_test2, X_val2, y_train2, y_test2, y_val2 = split_dataset(df2, y2)
# Splitting dataset into X_train, X_test, X_val, y_train, y_test, y_val for class3
X_train3, X_test3, X_val3, y_train3, y_test3, y_val3 = split_dataset(df3, y3)

# Visualising dataset
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
# Storing output with 1 column for training data
y_train_single = y_single(y_train)

# Concatenating all the validation data
X_val = pd.concat([X_val1, X_val2, X_val3])
y_val = np.concatenate((y_val1, y_val2, y_val3))
# Storing output with 1 column for validation data
y_val_single = y_single(y_val)

# Concatenating all the test data
X_test = pd.concat([X_test1, X_test2, X_test3])
y_test = np.concatenate((y_test1, y_test2, y_test3))
# Storing output with 1 column for test data
y_test_single = y_single(y_test)

# defining epochs, learning_rate, beta, error_diff
epochs = 500
learning_rate = 0.001
beta = 1
error_diff = 0.000001
J = 3

# Training MLFFNN with 1 hidden layer
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(X_train, y_train, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val_single)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Accuracy for validation data is : ", accuracy_val)
print("Confusion Matrix for training data : \n", conf_mat_val)

# predictions for test data
predictions_test = class_prediction_1hidden(X_test, wij, wjk, beta)
# Confusion matrix for test data
conf_mat_test = confusion_matrix(predictions_test, y_test_single)
# accuracy for test data
accuracy_test = accuracy(conf_mat_test)

print("Accuracy for test data is : ", accuracy_test)
print("Confusion Matrix for testing data : \n", conf_mat_test)

# predictions for training data
predictions_train = class_prediction_1hidden(X_train, wij, wjk, beta)
# Confusion matrix for training data
conf_mat_train = confusion_matrix(predictions_train, y_train_single)
# accuracy for training data
accuracy_train = accuracy(conf_mat_train)

print("Accuracy for training data is : ", accuracy_train)
print("Confusion Matrix for training data : \n", conf_mat_train)

# Plotting epoch error for the MLFFNN with one hidden layer
plt.figure()
plt.title("")
plt.scatter(epoch_num, epoch_error)
plt.xlabel("Epoch number")
plt.ylabel("Epoch Error")

# making decision plot with data visualisation
decision_plot_1hidden_MLFFNN(wij, wjk, beta, min_0, max_0, min_1, max_1)
plt.scatter(df1[1], df1[2], c='r')
plt.scatter(df2[1], df2[2], c='b')
plt.scatter(df3[1], df3[2], c='g')

# plotting output for each neuron
plot_output_1hidden(X_train, y_train_single, wij,wjk, beta)