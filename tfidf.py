from __future__ import division
from tokenizing import Tokenize
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
        def tokenize(sent): return self.preprocess.prepareToken(Tokenize, sent)

        return TfidfVectorizer(norm='l2',
                               min_df=0,
                               max_features=3000,
                               use_idf=True,
                               smooth_idf=False,
                               sublinear_tf=True,
                               tokenizer=tokenize)

    def fitData(self, yData):
        self.fullData = []
        for count, sent in enumerate(self.tfidf_data.toarray()):
            self.xData.append(sent)
            self.fullData.append([sent, yData[count]])

    def getOnlyXData(self):
        return self.xData

    def getFullData(self):
        return self.fullData

    def transform(self, sentence):
        factoryStem = StemmerFactory()
        stemmer = factoryStem.create_stemmer()

        factoryStop = StopWordRemoverFactory()
        stopper = factoryStop.create_stop_word_remover()

        sentStemmed = stemmer.stem(sentence)

        sentStopped = sentStemmed

        temp = stopper.remove(sentStemmed)
        while temp != sentStopped:
            sentStopped = temp
            temp = stopper.remove(sentStopped)

        return self.tfidf_vectorizer.transform([sentStopped]).toarray()[0]
