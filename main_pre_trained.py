from analiser import Analiser

an = Analiser()

an.load_model(file_name='model_5')

while True:
    sentence = input()
    print(an.testFromTrained([an.tfidf_data.transform(sentence)]))