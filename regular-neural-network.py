import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import timeit
import time

populationSize = 10

class TimeCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.timeOfTraining = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epochStartTime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epochDuration = (time.time() - self.epochStartTime)
        self.timeOfTraining += self.epochDuration
        self.times.append(self.epochDuration)

class RegularNeuralNetwork:
    batch_size = 128
    epochs = 8
    num_classes = 10

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Data sets downloaded")

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Data formatted")


    def __init__(self, id):
        self.id = id
    
    def defineModelArchitecture(self):
        self.model = Sequential()
        self.model.add(Dense(units=16, activation='relu', input_shape=(784,)))
        self.model.add(Dense(units=15, activation='relu'))
        self.model.add(Dense(units=10, activation='softmax'))

    def modelSummary(self):
        self.model.summary()
    
    def buildModel(self):
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    def trainModel(self):
        # self.fp = 'weights.{epoch:02d}-{val_accu:.2f}.hdf5'
        self.fp = 'weights_best.hdf5'

        checkPointCallback = keras.callbacks.callbacks.ModelCheckpoint(
            filepath = self.fp, 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=True, 
            mode='auto', 
            period=1)

        self.timeCallback = TimeCallback()

        callbacksList = [checkPointCallback, self.timeCallback]

        self.history = self.model.fit(
            self.x_train, 
            self.y_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            verbose=0, 
            validation_split=0.1,
            callbacks=callbacksList)

        self.model.load_weights(self.fp)

        self.timeOfTraining = self.timeCallback.timeOfTraining

    def evaluateModel(self):
        self.loss, self.accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)

    def results(self):
        print('Test loss:', self.loss)
        print('Test accuracy:', self.accuracy)
        # print('Training duration : ', self.trainingDuration)
        print('Training duration : ', self.timeOfTraining, 's')


    
    def saveModelToJSON(self):
        self.model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(self.model_json)

        self.model.save_weights("model.h5")
        print("model saved")

    def modelAccuracyPlot(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='best')
        plt.show()

    def modelLossPlot(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def peakBestWeights(self):
        pass



def sortKey(e): 
    return e.accuracy


def main():
    population = []
    for i in range(populationSize):
        rnn = RegularNeuralNetwork(i)
        rnn.defineModelArchitecture()
        rnn.buildModel()
        rnn.trainModel()
        rnn.evaluateModel()
        rnn.results()
        population.append(rnn)

    population.sort(reverse = True, key=sortKey)

    print('Best model no.', population[0].id, ' : ')
    population[0].results()


    

    
if __name__ == "__main__":
    main()
        



