from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from MLFFNN_linear_activation import feedforward_1hidden
import matplotlib.pyplot as plt

def plot_neurons_reg(X, wij,wjk, beta):
  """
    Input:
        X: Input data
        wij: weight matrix from input to hidden layer
        wjk: weight matrix from hidden to output layer
        beta: Parameter for sigmoid function
    This function plots the output of each neuron.
  """
  # converting X to numpy array
  X = np.array(X)
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
    ax.scatter3D(X[:,1], X[:,2],snj_array[:,i],c='r')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "Hidden Neuron"+str(i)
    ax.set_zlabel(label)
    plt.savefig(label)

  # plotting output layer outputs
  for i in range(len(snk_array[0])):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,1], X[:,2],snk_array[:,i], c='r')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    label = "Output Neuron"+str(i)
    ax.set_zlabel(label)
    plt.savefig(label)