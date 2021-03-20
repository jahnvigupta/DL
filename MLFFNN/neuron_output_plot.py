import numpy as np
from MLFFNN import feedforward_1hidden
from MLFFNN import feedforward_2hidden
import matplotlib.pyplot as plt

def plot_output_1hidden(X, y, wij,wjk, beta):
  """
    Input:
        X: Input data
        y: actual Output
        wij: weight matrix from input to hidden layer
        wjk: weight matrix from hidden to output layer
        beta: Parameter for sigmoid function
    This function plots the output of each neuron.
  """
  # converting X to numpy array
  X = np.array(X)

  # storing index of different classes
  index1 = []
  index2 = []
  index3 = []
  for i in range(len(X)):
    if(y[i]==0):
      index1.append(i)
    elif(y[i]==1):
      index2.append(i)
    else:
      index3.append(i)

  snj_array = []
  snk_array = []
  # computing feedforward part for all training examples
  for i in range(len(X)):
    anj, snj, ank, snk = feedforward_1hidden(X[i],wij,wjk,beta)
    snj_array.append(snj)
    snk_array.append(snk)

  snj_array = np.array(snj_array)
  snk_array = np.array(snk_array)

  # Plotting the hidden layer ouputs
  for i in range(len(snj_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[index1][:,1], X[index1][:,2],snj_array[index1][:,i],c='r')
    ax.scatter3D(X[index2][:,1], X[index2][:,2],snj_array[index2][:,i],c='b')
    ax.scatter3D(X[index3][:,1], X[index3][:,2],snj_array[index3][:,i],c='g')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "Hidden Neuron"+str(i)
    ax.set_zlabel(label)
    plt.savefig(label)

  # plotting output layer outputs
  for i in range(len(snk_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[index1][:,1], X[index1][:,2],snk_array[index1][:,i], c='r')
    ax.scatter3D(X[index2][:,1], X[index2][:,2],snk_array[index2][:,i], c='b')
    ax.scatter3D(X[index3][:,1], X[index3][:,2],snk_array[index3][:,i], c='g')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "Output Neuron"+str(i)
    ax.set_zlabel('label')
    plt.savefig(label)

def plot_output_2hidden(X, y, wij1, wj1j2, wj2k, beta):
  """
    Input:
        X: Input data
        y: actual Output
        wij1: weight vector from input layer to 1st hidden layer
        wj1j2: weight vector from 1st hidden layer to 2nd hidden layer
        wj1j2: weight vector from 2nd hidden layer to output layer
        beta: parameter of sigmoid function
    This function plots the output of each neuron.
  """
  # converting X to numpy array
  X = np.array(X)

  # storing index of different classes
  index1 = []
  index2 = []
  index3 = []
  for i in range(len(X)):
    if(y[i]==0):
      index1.append(i)
    elif(y[i]==1):
      index2.append(i)
    else:
      index3.append(i)

  snj1_array = []
  snj2_array = []
  snk_array = []
  # computing feedforward part for all training examples
  for i in range(len(X)):
    anj1, snj1, anj2, snj2, ank, snk = feedforward_2hidden(X[i],wij1,wj1j2,wj2k,beta)
    snj1_array.append(snj1)
    snj2_array.append(snj2)
    snk_array.append(snk)

  snj1_array = np.array(snj1_array)
  snj2_array = np.array(snj2_array)
  snk_array = np.array(snk_array)

  # Plotting the 1st hidden layer ouputs
  for i in range(len(snj1_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[index1][:,1], X[index1][:,2],snj1_array[index1][:,i],c='r')
    ax.scatter3D(X[index2][:,1], X[index2][:,2],snj1_array[index2][:,i],c='b')
    ax.scatter3D(X[index3][:,1], X[index3][:,2],snj1_array[index3][:,i],c='g')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "1st Hidden Neuron"+str(i)
    ax.set_zlabel(label)
    plt.savefig(label)
  
  # Plotting the 2nd hidden layer ouputs
  for i in range(len(snj2_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[index1][:,1], X[index1][:,2],snj2_array[index1][:,i],c='r')
    ax.scatter3D(X[index2][:,1], X[index2][:,2],snj2_array[index2][:,i],c='b')
    ax.scatter3D(X[index3][:,1], X[index3][:,2],snj2_array[index3][:,i],c='g')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "2nd Hidden Neuron"+str(i)
    ax.set_zlabel(label)
    plt.savefig(label)

  # plotting output layer outputs
  for i in range(len(snk_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[index1][:,1], X[index1][:,2],snk_array[index1][:,i], c='r')
    ax.scatter3D(X[index2][:,1], X[index2][:,2],snk_array[index2][:,i], c='b')
    ax.scatter3D(X[index3][:,1], X[index3][:,2],snk_array[index3][:,i], c='g')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "Output Neuron"+str(i)
    ax.set_zlabel('label')
    plt.savefig(label)
