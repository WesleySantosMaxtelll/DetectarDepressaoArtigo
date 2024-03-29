from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


class Models:
    ### Classifiers
    def Create_Classifier(self, name):
        if (name == 'LogReg'):
            return linear_model.LogisticRegression(solver='lbfgs', n_jobs=1, C=100)
        if (name == 'Baseline'):
            return DummyClassifier(strategy="most_frequent", random_state=None, constant=None)
        elif (name == 'KNN'):
            return KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
        elif (name == 'NB'):
            return MultinomialNB(alpha=0.1)
        elif (name == 'MLP'):
            # return MLPClassifier(hidden_layer_sizes=(30, 30, 30, 30, 30, 30, 30), alpha=1e-5, max_iter=500,  n_iter_no_change=50,
            #                      learning_rate_init=0.05, power_t=0.1, learning_rate='constant',  random_state=1)
            return MLPClassifier(learning_rate_init=0.005, verbose=True, tol=1e-6,
                                 hidden_layer_sizes=(20, 10, 10, 10, 10, 10, 10), alpha=0.01)
        else:
            raise NameError('Classifier Unavailable')

    def get_classifier(self, name):
        return self.Create_Classifier(name)




