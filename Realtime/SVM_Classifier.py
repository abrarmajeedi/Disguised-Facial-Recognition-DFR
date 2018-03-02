import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB



#train
df = pd.read_csv('anglesratiostrain.csv', header = None)
X_train = df.iloc[:,0:94]
y_train = df.iloc[:,-1:]
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)
y_train = y_train.reshape(1,-1)

scaler_tr= StandardScaler(with_mean = False).fit(X_train)
X_train = scaler_tr.transform(X_train)
scalery_tr = StandardScaler(with_mean = False).fit(y_train)
y_train = scalery_tr.transform(y_train)

y_train = np.array(y_train).T

#test
df = pd.read_csv('anglesratiostest.csv', header = None)
X_test = df.iloc[:,0:94]
y_test = df.iloc[:,-1:]
enc.fit(y_test)
y_test = enc.transform(y_test)
y_test = y_test.reshape(1,-1)

scaler_te= StandardScaler(with_mean = False).fit(X_test)
X_test = scaler_te.transform(X_test)
scalery_te = StandardScaler(with_mean = False).fit(y_test)
y_test = scalery_te.transform(y_test)
y_test = np.array(y_test).T
print "y done"
#predictions

df = pd.read_csv('anglesratiospredictions.csv', header = None)
X_pred = df.iloc[:,0:94]
y_pred = df.iloc[:,-1:]
enc.fit(y_pred)
y_pred = enc.transform(y_pred)
y_pred = y_pred.reshape(1,-1)

scaler_pr= StandardScaler(with_mean = False).fit(X_pred)
X_pred = scaler_te.transform(X_pred)
scalery_pr = StandardScaler(with_mean = False).fit(y_pred)
y_pred = scalery_te.transform(y_pred)
y_pred = np.array(y_pred).T
print y_pred
'''
model = KNeighborsClassifier(n_neighbors=28)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(predicted)
accuracy = accuracy_score(y_test, predicted)
print('test acc.:',accuracy)
pred_acc = accuracy_score(y_pred, model.predict(X_pred))
print('pred. acc.:',pred_acc)
'''

model = SVC(kernel = 'rbf', C = 1)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print('test acc.:',accuracy*100,'%')
pred_acc = accuracy_score(y_pred, model.predict(X_pred))
print('pred. acc.:',pred_acc*100,'%')










