from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read(file_name):
  df = pd.read_csv(file_name, skiprows=[1], header=None, sep=" ")
  df = df[[0,1]]
  return df

def split_dataset(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
  return X_train, X_test, X_val, y_train, y_test, y_val

X = read("NLS_Group09.txt")
len1 = 500
len2 = 500
len3 = 700
y1 = 0*np.ones(len1)
y2 = 1*np.ones(len2)
y3 = 2*np.ones(len3)

y = np.concatenate((y1,y2))
y = np.concatenate((y,y3))

X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X,y)