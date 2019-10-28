import re
from nltk.corpus import stopwords
import random

palavras = stopwords.words('portuguese')

def limpe_texto(textos, tags):
    textos_saida, tags_saida = [], []

    for t, l in zip(textos, tags):
        t = t.encode('ascii', 'ignore').decode('ascii')
        document = re.sub(r'\W', ' ', str(t))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        if re.search(r'\beu\b', document) and random.random() > 0.75:
            document = document.split()
            texto_final = ' '.join([d for d in document if d not in palavras])
            textos_saida.append(texto_final)
            tags_saida.append(l)


    return textos_saida, tags_saida


