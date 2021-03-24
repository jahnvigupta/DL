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

# storing train data images to file image_train.txt
train_file = open("image_train.txt", "w")
for row in bovw_train:
    np.savetxt(train_file, row)
train_file.close()

# storing test data images to file image_test.txt
test_file = open("image_test.txt", "w")
for row in bovw_test:
    np.savetxt(test_file, row)
test_file.close()