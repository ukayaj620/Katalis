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

    def __init__(self, training_data='dataset\processDatasets.csv'):
        self.preprocess(training_data)
        return None

    def preprocess(self, filepath):
        f = open(filepath)

        sentences = f.read().split('\n')
        sentences.pop(0)

        ff = open("check/before_shuffle.txt", 'w+')

        for sent in sentences:
            ff.write(sent)
            ff.write('\n')

        ff.close()

        shuffle(sentences)

        ff = open("check/after_shuffle.txt", 'w+')

        for sent in sentences:
            ff.write(sent)
            ff.write('\n')

        ff.close()

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

        json_file = open('model/' + file_name + '.json','r')
        loaded_model_json = json_file.read()

        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights('model/' + file_name + '.h5')
        print("Loaded model from disk")

        self.model_load = model
        return model

    def train(self, output_file='model'):
        x = self.tfidf_data.getOnlyXData()
        y = self.yData

        model = Sequential()

        input_data_dimen = len(x[0])
        input_data_dimen = 3000 if input_data_dimen > 3000 else input_data_dimen

        model.add(Dense(
            units=int(0.4 * input_data_dimen),
            activation='tanh',
            input_dim=input_data_dimen
        ))

        model.add(Dense(
            units=int(0.05 * 0.4 * input_data_dimen),
            activation='tanh'
        ))

        model.add(Dense(
            units=1,
            activation='sigmoid'
        ))

        learning_rate = .01
        batch_size = 1
        loss_error = 'binary_crossentropy'
        epoch = 10

        sgd = SGD(lr=learning_rate)

        model.compile(optimizer=sgd, loss=loss_error, metrics=['accuracy'])

        seed = 1;

        x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y),
                                                            test_size=0.2, random_state=seed)

        history = model.fit(x=x_train, y=y_train,
                            validation_data=(x_test, y_test),
                            batch_size=batch_size,
                            nb_epoch=epoch)


        #for plotting model accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train, test'], loc='upper left')
        plt.show()

        #for plotting model loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train, test'], loc='upper left')

        plt.show()

        self.save_model(model, output_file)

    def getBinaryResult(self, x):
        print(x)
        return "FORMAL" if x >= 0.5 else "NON FORMAL"

    def testFromTrained(self, x):
        if self.model_load == 'None':
            print("Model tidak ditemukan!")
            exit(0)

        return self.getBinaryResult(self.model_load.predict_proba(np.array(x)))


