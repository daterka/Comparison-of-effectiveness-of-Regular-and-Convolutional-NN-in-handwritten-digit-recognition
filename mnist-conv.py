from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs = 8

# input image dimensions
img_rows, img_cols = 28, 28

# TRAINING AND TEST DATA DOWNLOADING
# the data, split between train and test sets

#mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()
print("Data sets downloaded")

# DATA FORMATTING
#Reshaping the array to 4-dims so that it can work with the Keras API

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (img_rows, img_cols, 1)

# Making sure that the values are float so that we can get decimal points after division

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value

x_train /= 255
x_test /= 255

#convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("Data formatted")

# DEFINING MODEL ARCHITECTURE

model = Sequential() 
# First hidden layer - Convolutional
model.add(Conv2D(1, kernel_size=(3,3), input_shape=input_shape)) 

model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) 

# Second hidden layer - Fully Connected with 200 neurons
model.add(Dense(72,activation = 'relu'))

# Random neurons dropout (30% of)
model.add(Dropout(0.3)) 

# Third Layer - Output Layer
model.add(Dense(10,activation="softmax"))

model.summary()

# MODEL BUILT AND TRAINING WITH TRAINING DATASET
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose=1, validation_split=0.1)

#model.evaluate(x_test, y_test)

# ACCUARENCY CHECK WITH FEW TEST DATASET ELEMENTS

# score = model.evaluate(x_test, y_test, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~n')

print('Test loss:', loss)
print('Test accuracy:', accuracy)

# MODEL AND WEIGHTS SAVE

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("model.h5")
# print("model saved")

# model.summary()

# print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
