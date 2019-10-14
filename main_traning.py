from analiser import Analiser

an = Analiser()

an.train(output_file='model_1')

test_1 = "gw sk mkn"
print(an.testFromTrained([an.tfidf_data.transform([test_1])]))
