import pandas
import pickle
import re

def dados():
    path = '../dados_depressao/no_msg.P'

    def findWholeWord(w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    with open(path, 'rb') as fp:
        df = pickle.load(fp)
        return list(df.Text), list(df.Class)
        # print(df)
        # textos_brutos, tags_brutos = list(df.Text), list(df.Class)
        # print(len(textos_brutos))
        # textos, tags = [], []
        # for t, l in zip(textos_brutos, tags_brutos):
        #     if " eu " in t.lower():
        #         textos.append(t.lower())
        #         tags.append(l)
        #
        # print(len(textos))
        # print(len(tags))


# a, b = dados()
#
# print(len(a))
# print(len(b))