def split_dataset(X, y):
  """
    Input:
        X: Data values
        y: Actual Class of data values

    Output: 
        Data split into training, test and validation
  """
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
  return X_train, X_test, X_val, y_train, y_test, y_val