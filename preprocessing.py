import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factoryStem = StemmerFactory()
stemmer = factoryStem.create_stemmer()

factoryStop = StopWordRemoverFactory()
stopper = factoryStop.create_stop_word_remover()

xData = []
yData = []

rawDatasets = pd.read_csv('dataset\dataset_pool.csv', delimiter=',')

count = 0

for k in rawDatasets['Kalimat']:
    if count % 100 == 0:
        print(count)
    sentStemmed = stemmer.stem(k)

    sentStopped = sentStemmed

    '''
    temp = stopper.remove(k)

    while temp != sentStopped:
        sentStopped = temp
        temp = stopper.remove(sentStopped)
    '''


    xData.append(sentStopped)
    count += 1

for k in rawDatasets['Formalitas']:
    yData.append(k)

processDatasets = { 'Kalimat' : xData,
                    'Formalitas' : yData
                    }

dataFrameCSV = pd.DataFrame(processDatasets, columns=['Kalimat', 'Formalitas'])

dataFrameCSV.to_csv('D:\BIOS-Hackaton\Katalis\dataset\processedDataset_pool2.csv', index=None, header=True)

print(dataFrameCSV)
