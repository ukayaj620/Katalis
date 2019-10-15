from analiser import Analiser

# FIRST_MODEL pool1 0.005 2 20
# SECOND_MODEL pool2 0.005 2 20
# THIRD_MODEL pool2 0.01 16 25
# FOURTH_MODEL pool2 0.01 16 40
# FIFTH_MODEL pool2 0.01 32 45
# SIXTH_MODEL pool2 0.02 16 25
# SEVENTH_MODEL pool2 0.025 16 20
# EIGHTH_MODEL pool2 0.025 16 20 softmax


an = Analiser()

an.load_model(file_name='EIGHT_MODEL')

while True:
    sentence = input()
    print(an.testFromTrained([an.tfidf_data.transform(sentence)]))