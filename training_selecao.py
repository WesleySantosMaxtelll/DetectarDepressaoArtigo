
import numpy as np
import gensim
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
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
from sklearn.model_selection import train_test_split
from pickle import dump, load


def Classify(X, Y, cls, rep, k=5000):
    # Start moment
    Start_moment = time.time()
    title = 'Classificando com {} e {} k={}'.format(cls, rep, k)
    print(title)

    # Creating the K-fold cross validator
    if 'w2v' in rep:
        train_x = load(open('w2v_rep/{}_train_x.pkl'.format(rep), 'rb'))
        train_y = load(open('w2v_rep/{}_train_y.pkl'.format(rep), 'rb'))
        test_x = load(open('w2v_rep/{}_test_x.pkl'.format(rep), 'rb'))
        test_y = load(open('w2v_rep/{}_test_y.pkl'.format(rep), 'rb'))
    else:

        X_train, X_test, y_train, y_test =  train_test_split(X, Y,test_size=0.2,random_state=123,stratify=Y)
        train_x, train_y, test_x, test_y = Representations().get_representation(rep=rep, train_x=X_train, train_y=y_train,
                                                                                test_x=X_test, test_y=y_test, k=k, cat=None)
        sm = SMOTE(sampling_strategy='minority',
                       random_state=None)
        train_x, train_y = sm.fit_sample(train_x, train_y)

    # dump(train_x, open('w2v_rep/{}_train_x.pkl'.format(rep), 'wb'))
    # dump(train_y, open('w2v_rep/{}_train_y.pkl'.format(rep), 'wb'))
    # dump(test_x, open('w2v_rep/{}_test_x.pkl'.format(rep), 'wb'))
    # dump(test_y, open('w2v_rep/{}_test_y.pkl'.format(rep), 'wb'))
    # return

    classifier = Models().get_classifier(cls)
    classifier.fit(train_x, train_y)
        # Train_Classifier(classifier, X_train, Y_train)

    pred = classifier.predict(test_x)

    # report = classification_report(test_labels, test_pred, target_names=['Contrário', 'Favorável'] if plb =='polaridade' else ['neutro', 'opiniao'])
    report = classification_report(test_y, pred,
                                   target_names=['no','yes'])
    print(report)
    Finish_moment = time.time()
    tm = "It took " + str((Finish_moment - Start_moment)) + " seconds"
    print(tm)


classificadores = ['MLP']
representacao = ['w2v_mean', 'w2v']

textos, tags = dados()
print(len(textos))

textos, tags = limpe_texto(textos, tags)


print(len(textos))
print(len([t for t in tags if t ==0]))

# X_train, X_validacao, y_train, y_validacao = train_test_split(textos, tags,test_size=0.2,
#     random_state=123,stratify=tags)

for k in range(5000, 6000, 1000):
    for c in classificadores:
        for r in representacao:
            # Classify(X_train, y_train, c, r, k)
            Classify(textos, tags, c, r, k)
