from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def read(file_name):
  df = pd.read_csv(file_name, header=None, sep=" ")
  return df

def split_dataset(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
  return X_train, X_test, X_val, y_train, y_test, y_val

class1 = read("Class1.txt")
class2 = read("Class2.txt")
class3 = read("Class3.txt")
len1 = len(class1)
len2 = len(class2)
len3 = len(class3)
y1 = 0*np.ones(len1)
y2 = 1*np.ones(len2)
y3 = 2*np.ones(len3)

X = pd.concat([class1, class2], ignore_index=True)
X = pd.concat([X,class3], ignore_index=True)
y = np.concatenate((y1,y2))
y = np.concatenate((y,y3))

X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(X,y)