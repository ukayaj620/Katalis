from tfidf import TFIDF
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from random import shuffle
from keras.models import Sequential, model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Analiser:

    xData = []
    yData = []

    def __init__(self, training_data='dataset\processeddata.csv'):
        self.preprocess(training_data)
        return None

    def preprocess(self, filepath):
        f = open(filepath)

        sentences = f.read().split('\n')
        sentences.pop(0)

        for sentence in sentences:
            temp = sentence.split(',')
            if len(temp) == 2:
                self.xData.append(temp[0])
                self.yData.append(temp[1])

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
            if i == "1.0":
                y.append([1, 0])
            else:
                y.append([0, 1])

        model = Sequential()

        input_data_dimen = len(x[0])
        input_data_dimen = 500 if input_data_dimen > 500 else input_data_dimen

        model.add(Dense(
            units=int(0.4 * input_data_dimen),
            activation='tanh',
            input_dim=input_data_dimen
        ))

        model.add(Dense(
            units=int(0.05*0.4*input_data_dimen),
            activation='tanh'
        ))

        model.add(Dense(
            units=2,
            activation='softmax'
        ))

        learning_rate = .001
        batch_size = 1
        loss_error = 'categorical_crossentropy'
        epoch = 50

        sgd = SGD(lr=learning_rate)

        model.compile(optimizer=sgd, loss=loss_error, metrics=['accuracy'])

        x_train, x_test, y_train, y_test = self.train_custom_split(x, y, 0.75)

        self.history = model.fit(x=x_train, y=y_train,
                                 validation_data=(x_test, y_test),
                                 batch_size=batch_size,
                                 nb_epoch=epoch)

        self.save_model(model, output_file)

    def train_custom_split(self, x, y, sr_train):
        dataset = []

        for i in range(len(y)):
            dataset.append([x[i],y[i]])
        
        shuffle(dataset)
        
        formal=[]
        informal=[]

        for i in range(len(dataset)):
            if dataset[i][1][0] == 0:
                informal.append(dataset[i])
            else:
                formal.append(dataset[i])

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        formal_len = len(formal)
        formal_rat = formal_len * sr_train
        inform_len = len(informal)
        inform_rat = inform_len * sr_train

        for i in range(formal_len):
            if i < formal_rat:
                x_train.append(formal[i][0])
                y_train.append(formal[i][1])
            else:
                x_test.append(formal[i][0])
                y_test.append(formal[i][1])

        for i in range(inform_len):
            if i < inform_rat:
                x_train.append(informal[i][0])
                y_train.append(informal[i][1])
            else:
                x_test.append(informal[i][0])
                y_test.append(informal[i][1])
        
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def getBinaryResult(self, x):
        print(x)
        return "FORMAL" if x[0][0] > x[0][1] else "NON FORMAL"

    def testFromTrained(self, x):
        if self.model_load == 'None':
            print("Model tidak ditemukan!")
            exit(0)

        return self.getBinaryResult(self.model_load.predict_proba(np.array(x)))
