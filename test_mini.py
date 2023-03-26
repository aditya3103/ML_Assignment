# Put batch_size as len(x_train) for stochastic
# as 32 for mini batch
# and 1 for batch


import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

def initialize_weights(dim):
  w = np.zeros_like(dim)
  b = 0
  return w, b

def sigmoid(z):
  return (1/ (1+np.exp(-z)))

def logloss(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  log_loss = y_true*np.log10(y_pred)
  log_loss += (1-y_true)*np.log10(1-y_pred)
  return -np.mean(log_loss)

def gradient_dw(x, y, w, b, alpha, N):
  '''In this function, we will compute the gardient w.r.to w ''' 
  dw = x * (y-sigmoid(np.dot(w.T,x)+b)) - ((alpha*w*w)/N)
  return dw

def gradient_db(x, y, w, b):
  '''In this function, we will compute gradient w.r.to b ''' 
  db = y-sigmoid(np.dot(w.T,x)+b)
  return db

def train(X_train, y_train, X_test, y_test, epochs, alpha, eta0, batch_size):
  w, b = initialize_weights(X_train[0])
  N = len(X_train)
  log_loss_train = []
  log_loss_test = []

  for i in range(epochs):
    for j in range(0, N, batch_size):
      X_batch = X_train[j:j+batch_size]
      y_batch = y_train[j:j+batch_size]

      grad_dw = np.zeros_like(w)
      grad_db = 0
      for k in range(len(X_batch)):
        grad_dw += gradient_dw(X_batch[k], y_batch[k], w, b, alpha, batch_size)
        grad_db += gradient_db(X_batch[k], y_batch[k], w, b)

      w = np.array(w) + (eta0 * np.array(grad_dw))
      b = b + (eta0 * grad_db)

    # predict the output of x_train[for all data points in X_train] using w and b
    predict_train = []
    for m in range(len(y_train)):
        z = np.dot(w, X_train[m])+b
        predict_train.append(sigmoid(z)) 
    
    # store all the train loss values in a list
    train_loss = logloss(y_train, predict_train)

    # predict the output of x_test[for all data points in X_test] using w,b
    predict_test = []
    for m in range(len(y_test)):
        z = np.dot(w, X_test[m])+b
        predict_test.append(sigmoid(z))
    
    # store all the test loss values in a list
    test_loss = logloss(y_test, predict_test)

    # we can also compare previous loss and current loss, 
    #if loss is not updating then stop the process and return w,b
    if log_loss_train and train_loss > log_loss_train[-1]: 
      return w, b, log_loss_train, log_loss_test 
    
    log_loss_train.append(train_loss)
    log_loss_test.append(test_loss)

  return w, b, log_loss_train, log_loss_test




df = pd.read_csv('dataset.csv')
df = df.fillna(df.mean(numeric_only=True))

train_data = df.sample(frac=0.67, random_state=0)
test_data = df.drop(train_data.index)

x_train = train_data[train_data.columns[2:]].to_numpy()
y_train = train_data[train_data.columns[1]]
y_train.replace({'B': 0, 'M': 1}, inplace=True)
y_train = y_train.to_numpy()

x_test = test_data[test_data.columns[2:]].to_numpy()
y_test = test_data[test_data.columns[1]]
y_test.replace({'B': 0, 'M': 1}, inplace=True)
y_test = y_test.to_numpy()

alpha  = 0
eta0   = 0.001
epochs = 1000
w, b, log_loss_train, log_loss_test = train(x_train, y_train, 
                                            x_test, y_test, epochs, 
                                            alpha, eta0,len(x_train))

print ("weight vector: ", w)
print ("Intercept: ", b)

y_pred = []
for i in range(len(y_test)):
    z = np.dot(w, x_test[i]) + b
    y_pred.append(round(sigmoid(z)))

# calculate accuracy as percentage of correctly predicted outcomes
accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
print("Accuracy:", accuracy)