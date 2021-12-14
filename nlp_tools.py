import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize , sent_tokenize
from nltk import pos_tag 
from nltk.stem import SnowballStemmer
from nltk.stem import wordnet
from nltk import ngrams
import string

import warnings
warnings.filterwarnings('ignore')

import re
from functools import reduce


def tokenizationre(text):
    text1 = text.lower()
    # remove special characters (re)
    text2 = re.sub(r'[^a-z0-9 ]','',text1)
    # convert into list of words 
    return text2.split(' ')

def tokenization_word(text):
    tokens = []
    for word in word_tokenize(text.lower()):
        if word not in string.punctuation:
            tokens.append(word)
    return tokens

lemma = wordnet.WordNetLemmatizer() # initilizing word net
def lemmatization_sentence(sentence):#,stopwords,stop=True):
    """
    if stop == True : it will remove stopwords
    or else if stop == False : it will not remove stopwords
    """
    #try:
    # computing parts of speech
    tokens = tokenization_word(sentence)
    tag_list = pos_tag(tokens,tagset=None)
    lema_sent =[] # initizaing empty list

    for token,pos_token in tag_list:
        #if token not in stopwords:

        if pos_token.startswith('V'): # verb
            pos_val = 'v'
        elif pos_token.startswith('J'): # adjective
            pos_val = 'a'
        elif pos_token.startswith('R'): # adverb
            pos_val = 'r'
        else:# any parts of speech except verb, adjective, adverb
            pos_val = 'n'

        lema_token = lemma.lemmatize(token,pos_val) # computing lematization
        lema_sent.append(lema_token) # append values in list

    return " ".join(lema_sent)
#except:
    #return None
    
def bigram(text):
    tokens = tokenization(text)
    bi =list(ngrams(tokens,2,pad_left=True,pad_right=True,
                left_pad_symbol='<sos>',right_pad_symbol='</sos>'))
    return bi
