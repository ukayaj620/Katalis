from tfidf import TFIDF
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from random import shuffle
from keras.models import Sequential, model_from_json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Analiser:

    xData = []
    yData = []

    def __init__(self, training_data='dataset\processed_pool.csv'):
        self.preprocess(training_data)
        return None

    def preprocess(self, filepath):
        dataset = pd.read_csv(filepath, delimiter=',')

        self.xData = []
        self.yData = []

        for k in dataset['Kalimat']:
            self.xData.append(k)

        for k in dataset['Formalitas']:
            self.yData.append(k)

        self.tfidf_data = TFIDF([self.xData, self.yData])

    def save_model(self, model, file_name='model'):
        self.model_load = model

        model_json = model.to_json()
        with open('model/' + file_name + '.json', 'w') as json_file:
            json_file.write(model_json)

        model.save_weights('model/' + file_name + '.h5')
        print("Save model to disk")

    def load_model(self, file_name='model'):
        model = Sequential()

        json_file = open('model/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()

        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights('model/' + file_name + '.h5')
        print("Loaded model from disk")

        self.model_load = model
        return model

    def train(self, output_file='model'):
        x = self.tfidf_data.getOnlyXData()
        y = []

        for i in self.yData:
            if i == 1:
                y.append([1, 0])
            else:
                y.append([0, 1])

        model = Sequential()

        input_data_dimen = len(x[0])
        input_data_dimen = 1800 if input_data_dimen > 1800 else input_data_dimen

        model.add(Dense(
            units=int(0.4 * input_data_dimen),
            activation='tanh',
            input_dim=input_data_dimen
        ))

        model.add(Dense(
            units=int(0.075*0.4*input_data_dimen),
            activation='tanh'
        ))

        model.add(Dense(
            units=2,
            activation='softmax'
        ))

        learning_rate = .01
        batch_size = 16
        loss_error = 'categorical_crossentropy'
        epoch = 20

        sgd = SGD(lr=learning_rate)

        model.compile(optimizer=sgd, loss=loss_error, metrics=['accuracy'])

        x_train, x_test, y_train, y_test = self.train_custom_split(x, y, 0.6)

        self.history = model.fit(x=x_train, y=y_train,
                                 validation_data=(x_test, y_test),
                                 batch_size=batch_size,
                                 nb_epoch=epoch)

        self.save_model(model, output_file)

    def train_custom_split(self, x, y, sr_train, test_ratio=0.2):
        dataset = []

        for i in range(len(y)):
            dataset.append([x[i], y[i]])

        shuffle(dataset)

        formal = []
        informal = []

        for i in range(len(dataset)):
            if dataset[i][1][0] == 0:
                informal.append(dataset[i])
            else:
                formal.append(dataset[i])

        x_train = []
        x_test = []
        x_test_temp = []
        y_train = []
        y_test = []
        y_test_temp = []

        formal_len = len(formal)
        formal_rat = formal_len * sr_train
        inform_len = len(informal)
        inform_rat = inform_len * sr_train

        for i in range(formal_len):
            if i < formal_rat:
                x_train.append(formal[i][0])
                y_train.append(formal[i][1])
            else:
                x_test_temp.append(formal[i][0])
                y_test_temp.append(formal[i][1])

        for i in range(inform_len):
            if i < inform_rat:
                x_train.append(informal[i][0])
                y_train.append(informal[i][1])
            else:
                x_test_temp.append(informal[i][0])
                y_test_temp.append(informal[i][1])

        test_len = len(y_test_temp)
        test_rat = test_ratio * test_len

        for i in range(test_len):
            if i >= test_rat:
                x_train.append(x_test_temp[i])
                y_train.append(y_test_temp[i])
            else:
                x_test.append(x_test_temp[i])
                y_test.append(y_test_temp[i])

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def getBinaryResult(self, x):
        print(x)
        return "FORMAL" if x[0][0] > x[0][1] else "NON FORMAL"

    def testFromTrained(self, x):
        if self.model_load == 'None':
            print("Model tidak ditemukan!")
            exit(0)

        return self.getBinaryResult(self.model_load.predict_proba(np.array(x)))

    def showPlot(self):
        history = self.history

        # for plotting model accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('training_pic/model_acc.png')
        plt.show()

        # for plotting model loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('training_pic/model_loss.png')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()
