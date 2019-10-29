from collections import defaultdict
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
# model = KeyedVectors.load_word2vec_format('/home/maxtelll/Documents/arquivos/cbow_s50.txt')

class TfidfEmbeddingVectorizer(object):
    def __init__(self):
        self.word2vec =  KeyedVectors.load_word2vec_format('/home/maxtelll/Documents/arquivos/cbow_s50.txt')
        self.word2weight = None
        self.dim = 50
            # len(self.word2vec.itervalues().next())
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])



class w2v:
    def __init__(self):
        self.t = TfidfEmbeddingVectorizer()

    def fit_transform(self, X, Y):
        self.t.fit(X, Y)
        return self.t.transform(X)

    def transform(self, X):
        return self.t.transform(X)
#
# X = [
#     'oi tudo bem',
#     'sim e com vc',
#     'eita nois',
#     'ta'
# ]
#
# Y = [0,0,1,1]
#
# a = t.fit(X, Y)
# X2 = [
#     'oi tudo bem',
#     'mais o que é isso nois',
# ]
#
# print(a)