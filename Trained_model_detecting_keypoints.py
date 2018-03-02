import pandas as pd
import cv2
import os
import numpy as np
import timeit
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

dim = 227

df2 = pd.read_csv('test.csv',header = None)
print "Program started"
X = np.stack([cv2.imread("testimages/"+str(img)) for img in df2.iloc[:,-1]]).astype(np.float)[:, :, :, np.newaxis]
print "Resizing done"
y = np.vstack(df2.iloc[:,:-1].values)
X_test = X / 255
y_test = y


print "Model Started"

X_test = X_test.reshape(500,dim,dim,3)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, batch_size = 50,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("model compiled")


arr = []
files = []
for file in os.listdir(os.getcwd()+"//testimages//"):
	files.append(file)
	img = cv2.imread(os.getcwd()+"//testimages//"+str(file))
	arr = np.append(arr, img)

#files loaded
arr = arr.reshape(500,dim,dim,3)
arr = arr/255
print("predict started")
pred = loaded_model.predict(arr, batch_size = 50, verbose = 0)
pred = pd.DataFrame(pred)
files = pd.DataFrame(files)
x = pd.concat([pred,files], axis = 1)
"""
pred.to_csv('output.csv', header = None, index = None) 
files.to_csv('names.csv', header = None, index = None) 
"""
x.to_csv('predictions.csv', header = None, index = None) 





