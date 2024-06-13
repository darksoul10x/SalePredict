import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

advertising = pd.DataFrame(pd.read_csv("./advertising.csv"))
def printbox(str):
    print("----------------------")
    print(str)
    print("----------------------")
    return 0
# printbox(advertising.head())
# printbox(advertising.info())
# printbox(advertising.isnull().sum())

# Outlier Analysis
# fig, axs = plt.subplots(4, figsize = (5,5))
# plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
# plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
# plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
# plt4 = sns.boxplot(advertising['Sales'],ax= axs[3])
# plt.tight_layout()

# sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
# sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)

X = advertising['TV'].to_numpy()
y = advertising['Sales'].to_numpy()
X = np.array([X]).T
y = np.array([y]).T


one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 300, 2)
y0 = w_0 + w_1*x0
# Drawing the fitting line 
# plt.plot(X.T, y.T, 'ro')     # data 
# plt.plot(x0, y0)               # the fitting line
# plt.axis([0, 300, 0, 25])
# plt.xlabel('TV')
# plt.ylabel('Sales')
# plt.show()
def mean(x):
    return sum(x)/len(x)

yhat = []
for i in X:
    yhat.append(w_0 + w_1*i[0])

a = b = j = 0
d = []
for i in y:
    d.append(i[0])
m = float(mean(d))
for i in d:
    a = a + (i-m)**2
    b = b + (i-yhat[j])**2
    j = j+1
Rsquared =1 - b/a
print("Rsquared =",Rsquared)
# Rsquared = 1 - sum((y.T - y0)^2)/sum(y.T - mean(y.T))

# print("R =", Rsquared)
