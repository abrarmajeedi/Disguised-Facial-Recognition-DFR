import cv2
import numpy as np
import os
import pandas as pd
import timeit
from get_params import *
import timeit
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.svm import SVC	


dim = 227
font = cv2.FONT_HERSHEY_SIMPLEX
artrain = pd.read_csv('anglesratiostrain.csv', header = None)
X_train = artrain.iloc[:,0:94]
y_train = artrain.iloc[:,-1:]
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)
y_train = y_train.reshape(1,-1)

scaler = StandardScaler(with_mean = False).fit(X_train)
X_train = scaler.transform(X_train)

y_train = np.array(y_train).T

svc = SVC(kernel = 'rbf', C = 4)
svc.fit(X_train, y_train)
print('Classifier model fit')

#load regression model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
print("model compiled")


def get_text(params):
	x_svc = scaler.transform(params)
	y_svc = svc.predict(x_svc)
	y_svc = enc.inverse_transform(y_svc)
	y_svc = pd.DataFrame(y_svc)
	return y_svc.values.max()

arr = []
files = []

cap = cv2.VideoCapture('video.mp4')


# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
count  = 0
text = "loading"
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)	
    if ret == True:
	img2 = img[10:710, 290:990]
	new_img = cv2.resize(img2, (227, 227), interpolation = cv2.INTER_LINEAR)	
	count+=1
	arr = np.append(arr, new_img)
	if count%50 == 0:
		arr = arr.reshape(50,dim,dim,3)
		arr = arr/255
		print("predict started")
		pred = regressor.predict(arr, batch_size = 50, verbose = 0)
		pred = pd.DataFrame(pred)
		params = get_params(pred)
		text =  get_text(params)
		files = []
		arr = []
		count = 0
		cv2.imshow('Frame',frame)
		print text
		cv2.putText(frame,text, (100,100), font, 20, cv2.LINE_AA)
		
        # Display the resulting frame
        else:
		cv2.imshow('Frame',frame)
		cv2.putText(frame,text, (100,100), font, 20, cv2.LINE_AA)
		

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



