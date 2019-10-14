from analiser import Analiser

an = Analiser()

an.train(output_file='model')

while True:
    sentence = input()
    print(an.testFromTrained([an.tfidf_data.transform(sentence)]))

