from sklearn.model_selection import train_test_split
from MLFFNN import mlffnn_1hidden_layer
from class_predic_MLFFNN import class_prediction_1hidden
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from eval_param import confusion_matrix
from eval_param import accuracy

# Bovw process
# Importing libraries
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# To store 24-dimentional feature vectors of histogram. 
final_hist= []

# To store bovw representation of images and their class
# 33rd column in bovw_repr will be class as:
# 0: auditorium
# 1: desert_vegetation
# 2: synagogue_outdoor

bovw_repr= []
# RGB colors and their codes 
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

# To divide image into patch and get histogram
# Address of image is passed as parameter
def get_histo(address):
    # Open the image
    img = Image.open(address)
    # Convert the image into array
    img_array = np.array(img)
    # Find number of rows
    rows=img_array.shape[0]
    # Find number of columns
    cols=img_array.shape[1]
    # To make number of columns divisible by 32
    cols=32-cols%32
    # To copy pixels so that patches can be of size 32*32
    image_array_copy=[]
    for i in range(0,rows):
        a=[]
        for j in range(0,img_array.shape[1]):
            a.append(img_array[i][j])
        for j in range(0,cols):
            a.append(img_array[i][j])
        image_array_copy.append(a)
    # To make number of columns divisible by 32
    for i in range(32-rows%32):
        image_array_copy.append(image_array_copy[i])  
    # Divide the image into patches
    image_patches=[]
    for i in range(0,len(image_array_copy),32):
        for j in range(0,len(image_array_copy[0]),32):
            tmp=[]    
            for k in range(0,32):
                tmp1=[]
                for l in range(0,32):
                    tmp1.append(image_array_copy[i+k][j+l])
                tmp.append(tmp1)
            image_patches.append(tmp)
    image_patches=np.array(image_patches)
    # To generate histogram of patches
    for k in range(image_patches.shape[0]):
        patch_hist = []
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image_patches[k][ :, channel_id], bins=8, range=(0, 256)
            )
            # Concatenate 3, 8 dimensional feature vector into 1 24 dimensional feature vectors
            for i in range(len(histogram)):
                patch_hist.append(histogram[i])
        final_hist.append(patch_hist)

# To predict cluster of image
# Address of image is passed as parameter
def bovw(address, original_class):
    # Open the image
    img = Image.open(address)
    # Convert the image into array
    img_array = np.array(img)
    # Find number of rows
    rows=img_array.shape[0]
    # Find number of columns
    cols=img_array.shape[1]
    # To make number of columns divisible by 32
    cols=32-cols%32
    # To copy pixels so that patches can be of size 32*32
    image_array_copy=[]
    for i in range(0,rows):
        a=[]
        for j in range(0,img_array.shape[1]):
            a.append(img_array[i][j])
        for j in range(0,cols):
            a.append(img_array[i][j])
        image_array_copy.append(a)
    for i in range(32-rows%32):
        image_array_copy.append(image_array_copy[i])
    # Divide the image into patches
    image_patches=[]
    for i in range(0,len(image_array_copy),32):
        for j in range(0,len(image_array_copy[0]),32):
            tmp=[]    
            for k in range(0,32):
                tmp1=[]
                for l in range(0,32):
                    tmp1.append(image_array_copy[i+k][j+l])
                tmp.append(tmp1)
            image_patches.append(tmp)
    image_patches=np.array(image_patches)
    # To generate histogram of patches 
    patches = []
    for k in range(image_patches.shape[0]):
        patch_hist = []
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image_patches[k][ :, channel_id], bins=8, range=(0, 256)
            )
            # Concatenate 3, 8 dimensional feature vector into 1 24 dimensional feature vectors
            for i in range(len(histogram)):
                patch_hist.append(histogram[i])
        patches.append(patch_hist)
    # Predict using the trained model
    y_means=kmeans.predict(patches)
    count_cluster = []
    for i in range(32):
        count_cluster.append(0)
        count_cluster[i]=len(y_means[y_means==i])/len(y_means)
    if(original_class == "auditorium"):
        count_cluster.append(0)
    if(original_class == "desert_vegetation"):
        count_cluster.append(1)
    if(original_class == "synagogue_outdoor"):
        count_cluster.append(2)
    bovw_repr.append(count_cluster)
    
# Function to load images
# First parameter is address of folder
# Second parameter will be patch if image is to be divided into patches and generate histogram
# else it will be image to predict the cluster
def load(folder,method):
    if(method=="patch"):
        for filename in os.listdir(folder):
            path = folder + "/" + filename
            for cat in os.listdir(path):
                addr=path + "/" + cat
                get_histo(addr)
    if(method=="image"):
        for filename in os.listdir(folder):
            path = folder + "/" + filename
            for cat in os.listdir(path):
                addr=path + "/" + cat
                bovw(addr,filename)

# Main program begins from here
# load images and create patches and histogram
load("../Group09/Classification/Image_Group09/train","patch")
load("../Group09/Classification/Image_Group09/test","patch")
# Initialise and train the model
kmeans = KMeans(n_clusters=32)
kmeans.fit(final_hist)
# load images and create patches and predict clusters
load("../Group09/Classification/Image_Group09/train","image")
# bovw representation for train data
bovw_train = bovw_repr
bovw_repr = []
load("../Group09/Classification/Image_Group09/test","image")
# bovw representation for test data
bovw_test = bovw_repr

def y_3columns(y):
  """
    Input:
      y: Output vector with actual class in a single column
    Output:
      y_col: Output vector with different columns for different classes
  """
  y_col = np.zeros((len(y),3))
  for i in range(len(y)):
    y_col[i][int(y[i])] = 1
  return y_col

# converting bovw data to numpy array
bovw_train = np.array(bovw_train)
bovw_test = np.array(bovw_test)

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

# defining epochs, learning_rate, beta, error_diff
epochs = 150
learning_rate = 0.001
beta = 1
error_diff = 0.000001

# Training MLFFNN with 1 hidden layer
J = 40
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(pd.DataFrame(X_train), y_train_col, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Accuracy for validation data is : ", accuracy_val)
print("Confusion Matrix for training data : \n", conf_mat_val)

# Training MLFFNN with 1 hidden layer
J = 64
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(pd.DataFrame(X_train), y_train_col, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Accuracy for validation data is : ", accuracy_val)
print("Confusion Matrix for training data : \n", conf_mat_val)

# Training MLFFNN with 1 hidden layer
J = 96
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(pd.DataFrame(X_train), y_train_col, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Accuracy for validation data is : ", accuracy_val)
print("Confusion Matrix for training data : \n", conf_mat_val)

# Training MLFFNN with 1 hidden layer
J = 1000
epoch_error, epoch_num, wij, wjk = mlffnn_1hidden_layer(pd.DataFrame(X_train), y_train_col, epochs, learning_rate, beta, error_diff, J)

# predictions for validation data
predictions_val = class_prediction_1hidden(X_val, wij, wjk, beta)
# Confusion matrix for validation data
conf_mat_val = confusion_matrix(predictions_val, y_val)
# accuracy for validation data
accuracy_val = accuracy(conf_mat_val)

print("Accuracy for validation data is : ", accuracy_val)
print("Confusion Matrix for training data : \n", conf_mat_val)

# predictions for test data
predictions_test = class_prediction_1hidden(X_test, wij, wjk, beta)
# Confusion matrix for test data
conf_mat_test = confusion_matrix(predictions_test, y_test)
# accuracy for test data
accuracy_test = accuracy(conf_mat_test)

print("Accuracy for test data is : ", accuracy_test)
print("Confusion Matrix for testing data : \n", conf_mat_test)

# predictions for training data
predictions_train = class_prediction_1hidden(X_train, wij, wjk, beta)
# Confusion matrix for training data
conf_mat_train = confusion_matrix(predictions_train, y_train)
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