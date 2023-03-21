import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

def normalization(col):
    return (col-col.mean())/col.std()

def perceptron(traindata):

    y = traindata[traindata.columns[1]]
    result = {'B': 1, 'M': -1}
    y.replace(result, inplace=True)
    y = y.to_numpy()

    x = traindata[traindata.columns[2:]]
    x = x.to_numpy()

    w = np.zeros(len(x[0]))
    eta = 1
    epochs = 20

    for t in range(epochs):
            for i in range(len(x)):
                if (np.dot(x[i], w)*y[i]) <= 0:
                    w = w + eta*x[i]*y[i]
    
    return w


# Data Preprocessing

df = pd.read_csv('dataset.csv')
df = df.fillna(df.mean(numeric_only=True))

df1 = df.copy()

for col in df.columns:
    if(col=='diagnosis'):
        continue 
    df[col] = normalization(df[col])

# Splitting Training and Testing Data


# Perceptron Algorithm



# Perceptron Model 1

w1 = perceptron(df1.sample(frac=0.67, random_state=333))
w2 = perceptron(df1.sample(frac=0.67, random_state=123))
w3 = perceptron(df.sample(frac=0.67, random_state=241))
w4 = perceptron(df[list(df.columns[:3]) + list(df.columns[np.random.permutation(np.arange(3, len(df.columns)))])].sample(frac=0.67, random_state=241))

print(w1)
print(w2)
print(w3)
print(w4)




# Perceptron Model 2

# Perceptron Model 3

# Perceptron Model 4



