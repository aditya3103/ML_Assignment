import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def normalization(col):
    return (col-col.mean())/col.std()

# assume Gaussian distribution for each class in 1D
def gaussian(x, mean, var):
    return 1/(np.sqrt(2*np.pi*var)) * np.exp(-((x-mean)**2)/(2*var))

# define generative model for decision boundary
def generative_model(x):
    pos_likelihood = gaussian(x, mean_pos, var_pos)
    neg_likelihood = gaussian(x, mean_neg, var_neg)
    if pos_likelihood > neg_likelihood:
        return 1
    else:
        return 0

df = pd.read_csv('dataset.csv')
df = df.fillna(df.mean(numeric_only=True))

def FLDM(x_train, y_train):
    flda = LinearDiscriminantAnalysis(n_components=1)
    flda.fit(x_train, y_train)

    X_train_1D = flda.transform(x_train)

    # find mean and variance of the 1D data for each class
    mean_pos = np.mean(X_train_1D[y_train==1])
    mean_neg = np.mean(X_train_1D[y_train==0])
    var_pos = np.var(X_train_1D[y_train==1])
    var_neg = np.var(X_train_1D[y_train==0])

    # find decision boundary by searching for the point where pos_likelihood = neg_likelihood
    # x = np.linspace(np.min(X_train_1D), np.max(X_train_1D), 1000)
    likelihood_diff = [gaussian(X_train_1D[i], mean_pos, var_pos) - gaussian(X_train_1D[i], mean_neg, var_neg) for i in range(len(X_train_1D))]
    decision_boundary = X_train_1D[np.argmin(np.abs(likelihood_diff))]

    # print decision boundary
    print("Decision boundary:", decision_boundary)


for col in df.columns:
    if(col=='diagnosis'):
        continue 
    df[col] = normalization(df[col])

train_data = df.sample(frac=0.67, random_state=23)
test_data = df.drop(train_data.index)

x_train = train_data[train_data.columns[2:]].to_numpy()
y_train = train_data[train_data.columns[1]]
y_train.replace({'B': 1, 'M': 0}, inplace=True)
y_train = y_train.to_numpy()

x_test = test_data[test_data.columns[2:]].to_numpy()
y_test = test_data[test_data.columns[1]]
y_test.replace({'B': 1, 'M': 0}, inplace=True)
y_test = y_test.to_numpy()

FLDM(x_train[:, np.random.permutation(len(x_train[0]))], y_train)
FLDM(x_train, y_train)









