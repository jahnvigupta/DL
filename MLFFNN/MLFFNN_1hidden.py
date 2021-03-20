def initialise_weights(J, K):
  """
    Input:
        J: Number of neurons in first layer
        K: Number of neurons in second layer
        
    Output: 
        w: random weight vector of dimension (J+1)*K
    This function initialises the weights of the Neural Network from first layer to second layer 
  """
  w = np.random.rand(J,K)
  return w

def feedforward_1hidden(x,wij,wjk,beta):
  """
    Input:
        x: Vector for which feedforward computation is to be done
        wij: weight matrix from input to hidden layer
        wjk: weight matrix from hidden to output layer
        beta: Parameter for sigmoid function
    Output: 
        anj: activation value from hidden layer
        snj: output from hidden layer
        ank: activation value from output layer
        snk: output from output layer
  """
  # Computation through hidden layer
  anj = wij.T.dot(x)
  snj = sigmoid_vector(anj, beta)
  snj_bias = np.append(1,snj) # snj with bias term

  # computation through output layer
  ank = wjk.T.dot(snj_bias)
  snk = sigmoid_vector(ank, beta)
  return anj, snj, ank, snk

def update_wjk_1hidden(yk, sk, ak, beta, sj_bias, learning_rate):
  """
    Input:
        yk: Actual prediction
        sk: Prediction made by network
        ak: activation value from output layer
        beta: parameter of sigmoid function
        sj_bias: input to output layer
        learning_rate: Learning Rate of Gradient Descent
        
    Output: 
        update: Update to wjk
  """
  # derivative of f(ak)
  d_f = beta*sk*(1-sk)
  delta_k = (yk-sk)*d_f
  update = learning_rate*np.outer(sj_bias,delta_k)
  return update

def update_wij_1hidden(x, yk, sk, ak, wjk, beta, sj, learning_rate):
  """
    Input:
        x: Input to Neural Network
        yk: Actual prediction
        sk: Prediction made by network
        ak: activation value through output layer
        wjk: Weight matrix from hidden to output layer
        beta: parameter of sigmoid function
        sj: output from hidden layer
        learning_rate: Learning Rate of Gradient Descent
        
    Output: 
        update: Update to wjk
  """
  d_fk = beta*sk*(1-sk)
  delta_k = (yk-sk)*d_fk
  temp = np.dot(delta_k,wjk[1:,:].T)
  d_fj = beta*sj*(1-sj)
  delta_j = temp*d_fj
  update = learning_rate*np.outer(x, delta_j)
  return learning_rate*update

def mlffnn_1hidden_layer(X_train, y_train, epochs, learning_rate, beta, error_diff, J):
  """
    Input:
        X_train: Training data points
        y_train: actual values for training points
        epochs: Number of epochs for training
        learning_rate: Learning rate for Gradient Descent
        beta: Parameter for sigmoid function
        error_diff: Threshold on error difference between successive epochs
        J: Number of neurons in hidden layer
    Output: 
        wij: Weight vector from input to hidden layer after training
        wjk: Weight vector from hidden to output layer after training
        epoch_num: List containing epoch numbers
        epoch_error: List containing errors on every epoch
  """
  # initialising empty list for epoch numbers and epoch error
  epoch_error = []
  epoch_num = []
  # i is iteration number
  i = 0

  # Initialising weights from input layer to hidden layer randomly
  wij = initialise_weights(len(X_train.columns),J)
  # Initialising weights from hidden layer output layer randomly
  wjk = initialise_weights(J+1,len(y_train[0]))
  
  # Converting X_train to numpy array
  X_train = np.array(X_train)
  # Running iteration through all the training points till difference between consecutive errors is above threshold
  while(len(epoch_num)<=2 or ((-epoch_error[i-1]+epoch_error[i-2])>error_diff or epoch_error[i-2]<epoch_error[i-1])):
    # Condition learning_rate on iteration number
    if(i<50):
      learning_rate = 0.1
    elif(i<200):
      learning_rate = 0.001
    else:
      learning_rate = 0.0001
    # Shuffle training data for each epoch
    X_train, y_train = shuffle(X_train, y_train)
    # predictions is used for storing the predictions of the model for training data
    predictions = []
    # Updating weight vector for every data point
    for j in range(len(X_train)):
      # Feedforward part
      anj, snj, ank, snk = feedforward_1hidden(X_train[j],wij,wjk,beta)
      snj_bias = np.append(1,snj)
      predictions.append(snk)

      # backpropogation
      wjk_copy = wjk
      wjk = wjk + update_wjk_1hidden(y_train[j], snk, ank, beta, snj_bias, learning_rate)
      wij = wij + update_wij_1hidden(X_train[j], y_train[j], snk, ank, wjk_copy, beta, snj, learning_rate)

    # epoch_num stores the epoch number
    epoch_num.append(i)
    # epoch_error stores error for each epoch
    epoch_error.append(avg_error(y_train, predictions))
    i = i+1
  return epoch_error, epoch_num, wij, wjk