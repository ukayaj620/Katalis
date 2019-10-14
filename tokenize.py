class Tokenize:

    def __init__(self):
        return None

    def tokenizing(self, sentence):
        token = sentence.split(' ')
        return token

    def prepareToken(self, sentences):
        return self.tokenizing(self, sentence=sentences)