import pandas as pd
import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers, callbacks
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K



dim = 227
tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#train
df1 = pd.read_csv('train.csv',header = None)
print "Program started"
X = np.stack([cv2.imread("trainimages/"+str(img)) for img in df1.iloc[:,-1]]).astype(np.float)[:, :, :, np.newaxis]
print "Resizing done"
y = np.vstack(df1.iloc[:,:-1].values)
X_train = X / 255
y_train = y


#test
df2 = pd.read_csv('test.csv',header = None)
print "Program started"
X = np.stack([cv2.imread("testimages/"+str(img)) for img in df2.iloc[:,-1]]).astype(np.float)[:, :, :, np.newaxis]
print "Resizing done"
y = np.vstack(df2.iloc[:,:-1].values)
X_test = X / 255
y_test = y


print "Model Started"

X_train = X_train.reshape(3500,dim,dim,3)
X_test = X_test.reshape(500,dim,dim,3)

print "X_train.shape" + str(X_train.shape)
print "y_train.shape" + str(y_train.shape)
print "X_test.shape" + str(X_test.shape)
print "y_test.shape" + str(y_test.shape)



model = Sequential()
model.add(BatchNormalization(batch_size = 50, input_shape=(dim,dim,3)))
model.add(Conv2D(8,kernel_size=(2,2),strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(Conv2D(8, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))


model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Conv2D(16, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(Conv2D(16, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(32, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(64, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(128, kernel_size=(5,5), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(256, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(256, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Conv2D(512, kernel_size=(2,2), strides=(1, 1), padding='same', data_format=None, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')) 

model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))


model.add(Dense(40))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=50, epochs=1200, verbose=1, validation_split=0.1, shuffle=True,callbacks = [tbCallBack])

#saving model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#evaluate model
scores = model.evaluate(X_test, y_test, verbose=1, batch_size = 50)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

