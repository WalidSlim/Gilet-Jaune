# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:18:50 2019

@author: PC
"""
import pandas as pd
import numpy as np  
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
import re #import the regular expressions library; will be used to strip punctuation

f=open('discours.txt','r')
data=f.read()
f.close()

#Cleaning:

words = word_tokenize(data)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

stopWords = set(stopwords.words('french'))

def remove_stop(words):
    wf = []
    for w in words:
      if w not in stopWords:
        wf.append(w)
    return wf 
     
#Prétraitement :
tl=to_lowercase(words)
new_words=remove_punctuation(tl)
rs=remove_stop(new_words)


def stem_words(words):
    
    #stemming words
    stemmed_words = [] 
    stemmer = FrenchStemmer() #creation d'un  objet stemmer  
                              #dans la classe FrenchStemmer 
    for word in words:
        stemmed_word=stemmer.stem(word) #stem the word
        stemmed_words.append(stemmed_word) 
    return stemmed_words

sw=stem_words(rs)

#frequence d'utilisation d'un mot
fdist = FreqDist(rs)

frequency_frame = pd.DataFrame(fdist.most_common(30),
                                        columns=["mots", "frequences"])


# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30)
plt.show()

#2 eme etape recuperer les mots les plus utilisés : 
print(fdist.most_common(30))



dico = {}
for key, value in fdist.most_common(50):
    if key not in dico:
        dico[key] = [value]
    else:
        dico[key].append(value)
print (dico) 

#POS Tagging
import os
java_path = "C:\\Program Files\\Java\\jdk1.8.0_121\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java(r"C:\\Program Files\\Java\\jdk1.8.0_121\\bin\\java.exe")
from nltk.tag.stanford import StanfordPOSTagger 
path_to_model = (r"C:\Users\acer\Desktop\Stanford\stanford-postagger-full-2018-10-16\models\french.tagger")
path_to_jar =(r"C:\Users\acer\Desktop\Stanford\stanford-postagger-full-2018-10-16/stanford-postagger.jar")
tagger = nltk.tag.stanford.StanfordPOSTagger(path_to_model, path_to_jar)

chn = " ".join(dico.keys()) 
# transformer ma liste rs (remove stops) en chaine

tokens = nltk.tokenize.word_tokenize(chn)
print (tagger.tag(tokens))

#print(list(nltk.bigrams(chn)))


dico2 = {}
for key, value in tagger.tag(tokens):
 
    if value not in dico2:
        dico2[value] = [key]
    else:
        dico2[value].append(key)
print (dico2) 

# je cherche la longueur maximale pour construire un DataFrame :    
l=[]
for i in dico2.values():
   l.append(len(i))
d=sorted(l)
print(d[len(l)-1]) 

maxx=d[len(l)-1]

for key in dico2:
    while(len(dico2[key])<maxx):
        dico2[key].append('rien')
print( dico2)

#d= np.array(tagger.tag(tokens))
df=pd.DataFrame.from_dict(dico2)
print(df)
#df.head() 

from textblob import TextBlob
from textblob_fr import PatternAnalyzer 

def sentiment_calc(text):
    blob=TextBlob(text,analyzer=PatternAnalyzer())
    return blob.sentiment
   

df['sentiment'] = df['ADJ'].apply(sentiment_calc)
#Pour voir chacun dans une colonne : 
df['Polarity'] = df['sentiment'].apply(lambda x: x[0])
df['Subjectivity'] = df['sentiment'].apply(lambda x: x[1])


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df['Subjectivity'],label="Count")
sns.countplot(df['Polarity'],label="Count")
plt.show()


def lexical_diversity(data): 
    return len(set(data)) / len(data)








