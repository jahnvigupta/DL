import numpy as np
from MLFFNN_linear_activation import feedforward_1hidden
import matplotlib.pyplot as plt

def plot_neurons_reg_uni(X, wij,wjk, beta):
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
    plt.figure()
    plt.scatter(X[:,1],snj_array[:,i],c='r')
    plt.xlabel('Feature 1')
    label = "Hidden Neuron"+str(i)
    plt.ylabel(label)
    plt.savefig(label)

  # plotting output layer outputs
  for i in range(len(snk_array[0])):
    plt.figure()
    plt.scatter(X[:,1],snk_array[:,i],c='r')
    plt.xlabel('Feature 1')
    label = "Output Neuron"+str(i)
    plt.ylabel(label)
    plt.savefig(label)