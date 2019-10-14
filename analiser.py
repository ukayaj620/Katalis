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

        shuffle(sentences)

        for sentence in sentences:
            temp = sentence.split(',')
            if len(temp) == 2:
                self.xData.append(temp[0])
                self.yData.append(temp[1])

        self.tfidf_data = TFIDF([self.xData, self.yData])