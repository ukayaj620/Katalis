from __future__ import division
from tokenize import Tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TFIDF:

    fullData = []
    xData = []

    def __init__(self, data):
        self.preprocess = Tokenize
        self.tfidf_vectorizer = self.initVectorizer()
        self.tfidf_data = self.tfidf_vectorizer.fit_transform(data[0])
        self.fitData(data[1])

    def initVectorizer(self):
        tokenize = lambda sent : self.preprocess.prepareToken(Tokenize, sent)

        return TfidfVectorizer(norm='l2',
                               min_df=0,
                               max_features='3000',
                               use_idf=True,
                               smooth_idf=False,
                               sublinear_tf=True,
                               tokenizer=tokenize)

    def fitData(self, yData):
        i = 0
        self.fullData = []
        for count, sent in enumerate(self.tfidf_data.toarray()):
            self.xData.append(sent)
            self.fullData.append([sent, yData[i]])
            i += 1

    def getOnlyXData(self):
        return self.xData

    def getFullData(self):
        return self.fullData
