from collections import defaultdict
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

class MeanEmbeddingVectorizer(object):
    def __init__(self):
        self.word2vec = KeyedVectors.load_word2vec_format('/home/maxtelll/Documents/arquivos/cbow_s50.txt')
        self.word2weight = None
        self.dim = 50


    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class w2v_mean:
    def __init__(self):
        self.t = MeanEmbeddingVectorizer()

    def fit_transform(self, X, Y):
        self.t.fit(X, Y)
        return self.t.transform(X)

    def transform(self, X):
        return self.t.transform(X)