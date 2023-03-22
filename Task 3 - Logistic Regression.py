import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

def normalization(col):
    return (col-col.mean())/col.std()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hypothesis(theta, x):
    return sigmoid(z(theta, x))

def z(theta, x):
    return np.dot(x, theta)

def cost(theta, x, y):
    h = hypothesis(theta, x)
    one_case = np.matmul(-y.T, np.log(h))
    zero_case = np.matmul(-(1 - y).T, np.log(1 - h))
    return (one_case + zero_case)/len()


def gradient_descent(theta, x, y, learning_rate, regularization = 0):
    dw = (2/len(x))*(np.matmul(x.T, hypothesis(theta, x)-y))
    return theta - learning_rate*dw

    

def minimize(theta, x, y, iterations, learning_rate, regularization = 0):
    for i in range(iterations):
        theta = gradient_descent(theta, x, y, learning_rate, regularization)
        #costs.append(cost(theta, x, y)[0][0])
    return theta

# Data Preprocessing

df = pd.read_csv('dataset.csv')
df = df.fillna(df.mean(numeric_only=True))

train_data = df.sample(frac=0.67, random_state=123)
test_data = df.drop(train_data.index)

x_train = train_data[train_data.columns[2:]].to_numpy()
y_train = train_data[train_data.columns[1]]
y_train.replace({'B': 0, 'M': 1}, inplace=True)
y_train = y_train.to_numpy()

print(y_train.shape)


#theta = minimize(np.zeros(len(x_train[11]),), x_train, y_train, 200, 1.2, 0.5)
#print((test_data[test_data.columns[2:]].to_numpy()))
#print(theta)
#print(np.dot(theta, (test_data[test_data.columns[2:]].to_numpy())[111]))
y_test = test_data[test_data.columns[1]]
y_test.replace({'B': 0, 'M': 1}, inplace=True)
y_test = y_test.to_numpy()
#print(y_test[1])





