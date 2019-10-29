import nltk as nltk
from string import punctuation
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from selecao_de_atributos import Selecao

from w2v import w2v
from w2v_media import w2v_mean


class Representations:

    def Create_Vectorizer(self, name, k, cat):

        if (name == 'CountVec'):
            return CountVectorizer(analyzer="word", stop_words=nltk.corpus.stopwords.words('portuguese'),
                                   max_features=5000)
        elif (name == 'NGram'):
            return CountVectorizer(analyzer="char", ngram_range=([3, 16]), tokenizer=None, preprocessor=None,
                                   max_features=3000)
        elif (name == 'TFidf'):
            return TfidfVectorizer(min_df=2, stop_words=nltk.corpus.stopwords.words('portuguese'))

        elif (name == 'selecao'):
            return Selecao(k, cat)

        elif (name == 'w2v'):
            return w2v()

        elif (name == 'w2v_mean'):
            return w2v_mean()
        else:
            raise NameError('Vectorizer not found')

    def get_representation(self, rep, train_x, train_y, test_x, test_y, k, cat):
        vec = self.Create_Vectorizer(rep, k, cat)

        X_train = vec.fit_transform(train_x, train_y)
        Y_train = np.array(train_y)
        # vec.mostre_melhores()
        X_test = vec.transform(test_x)
        Y_test = np.array(test_y)

        return X_train, Y_train, X_test, Y_test
