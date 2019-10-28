
import numpy as np
import gensim
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report
import gc
import scipy.sparse as sp
import time
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from representations import Representations
from Models import Models
from obter_dados import dados
from tratar_texto import limpe_texto


def Classify(X, Y, cls, rep, k=5000):
    # Start moment
    Start_moment = time.time()
    title = 'Classificando com {} e {} k={}'.format(cls, rep, k)
    print(title)

    # Creating the K-fold cross validator
    K_fold = KFold(n_splits=10, shuffle=True)

    # Labels
    test_labels = np.array([], 'int32')
    test_pred = np.array([], 'int32')

    # Confusion Matrix
    confusion = np.array([[0, 0], [ 0, 0]])

    # The test
    for train_indices, test_indices in K_fold.split(X):
        print('Running .... =)')
        X_train = [X[i] for i in train_indices]
        Y_train = [Y[i] for i in train_indices]

        X_test = [X[i] for i in test_indices]
        Y_test = [Y[i] for i in test_indices]

        train_x, train_y, test_x, test_y = Representations().get_representation(
            rep=rep, train_x=X_train, train_y=Y_train, test_x=X_test, test_y=Y_test, k=k, cat=None)
        # c = Counter(Y_train)
        # print(Counter(train_y))
        # print({1:c.most_common(1)[0][1], 0:c.most_common(1)[0][1], 2:c.most_common(1)[0][1]})

        sm = SMOTE(sampling_strategy='minority',
                   random_state=None)
        # sm = SMOTE(sampling_strategy={1:c.most_common(1)[0][1], 0:c.most_common(1)[0][1], 2:c.most_common(1)[0][1]}, random_state=None)
        # print(len(train_y))
        train_x, train_y = sm.fit_sample(train_x, train_y)

        # print(Counter(train_y))

        test_labels = np.append(test_labels, Y_test)

        classifier = Models().get_classifier(cls)
        classifier.fit(train_x, train_y)
        # Train_Classifier(classifier, X_train, Y_train)

        pred = classifier.predict(test_x)
        test_pred = np.append(test_pred, pred)
        # print(test_y)
        # print(pred)
        confusion += confusion_matrix(test_y, pred)

    # report = classification_report(test_labels, test_pred, target_names=['Contrário', 'Favorável'] if plb =='polaridade' else ['neutro', 'opiniao'])
    report = classification_report(test_labels, test_pred,
                                   target_names=['no','yes'])
    print(report)
    print("Confusion matrix:")
    print(confusion)
    Finish_moment = time.time()
    tm = "It took " + str((Finish_moment - Start_moment)) + " seconds"
    print(tm)
    #
    # f = open(Output_string(cls, clf), 'w+')
    # f.write('Word2vec with ' + clf + '\n Classifying ' + cls + '\n')
    # f.write(title + '\n \n')
    # f.write(report + '\n \n')
    # f.write("Confusion Matrix: \n")
    # f.write(np.array_str(confusion) + '\n \n')
    # f.write(tm)
    # f.close()

classificadores = ['LogReg', 'NB', 'MLP']
representacao = ['TFidf', 'CountVec']

textos, tags = dados()
textos, tags = limpe_texto(textos, tags)

print(len(textos))
print(len([t for t in tags if t =='no']))

for k in range(5000, 6000, 1000):
    for c in classificadores:
        for r in representacao:
            Classify(textos, tags, c, r, k)

