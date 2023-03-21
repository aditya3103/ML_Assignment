import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

def normalization(col):
    return (col-col.mean())/col.std()


# Data Preprocessing

df = pd.read_csv('dataset.csv')
df = df.fillna(df.mean(numeric_only=True))

for col in df.columns:
    if(col=='diagnosis'):
        continue 
    df[col] = normalization(df[col])

# Splitting Training and Testing Data

train = df.sample(frac=0.67, random_state=np.random.randint(1,200))
test = df.drop(train.index)

# Perceptron Model 1

# Perceptron Model 2

# Perceptron Model 3

# Perceptron Model 4



