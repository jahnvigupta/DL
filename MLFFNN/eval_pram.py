import numpy as np

def confusion_matrix(predic, truth):
  """
    Input:
        predic: Predictions made by the model
        truth: gound truth
        
    Output: 
        conf_mat: Confusion matrix  
  """
  conf_mat = np.zeros((3,3))
  for i in range(len(truth)):
    conf_mat[int(truth[i])][int(predic[i])] = conf_mat[int(truth[i])][int(predic[i])]+1
  return conf_mat

def accuracy(Confusion_matrix):
  """
    Input:
        Confusion_matrix: Confusion Matrix
        
    Output: 
        Accuracy
  """
  accuracy = 0
  for i in range(len(Confusion_matrix)):
    accuracy = accuracy + Confusion_matrix[i][i]
  total = 0
  for i in range(len(Confusion_matrix)):
    for j in range(len(Confusion_matrix)):
      total = total + Confusion_matrix[i][j]
  return (accuracy/total)*100