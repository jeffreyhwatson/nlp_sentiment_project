import string, re
import pandas as pd
import numpy as np
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def re_tokens(tweet):
    stop_list = stopwords.words('english')
    stop_set = set(stop_list)
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tokens = tokenizer.tokenize(tweet)
    no_stopwords = [token.lower() for token in tokens if
                    token.lower() not in stop_set]
    return no_stopwords

def vocabulary(data):
    vocab = [word for tweet in data for word in tweet.split()]
    return set(vocab)

def lemmatize(processed_data):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for doc in processed_data:
        lemms = ' '.join([lemmatizer.lemmatize(word) for word in doc])
        lemmas.append(lemms)
    return lemmas

def stemmatize(processed_data):
    ps = PorterStemmer()
    stemmed = []
    for doc in processed_data:
        stems = ' '.join([ps.stem(word) for word in doc])
        stemmed.append(stems)    
    return stemmed

def ht_extract(data):
    hashlists = []
    for element in data:
        hashtag = re.findall(r'\B#\w*[a-zA-Z]+\w*', element)
        hashlists.append(hashtag)
    hashtags = [hashtag.lower() for h_list in hashlists for hashtag in h_list]
    return hashtags

def find_strings(data, expression):
    strings = []
    for tweet in data:
        string = re.findall(expression, tweet)
        if len(string) != 0:
            strings.append(string)
    return strings

def string_checker(data_list, string):
    if string in data_list:
        print('string is in data')
    else:
        print('string is not in data')

def clean_lemms(data):
    stripped_data = []
    subs = [(r'\{link\}', ''),
            (r'http\S+', ''),
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('(&amp)', ''),
            ('(&quot)', ''),
            ('(&nbsp)', ''),
            ('(&lt)', ''),
            ('(&gt)', ''),
            ('(RT\s)', '')
           ]
    for tweet in data:
        for pair in subs:
            tweet = re.sub(pair[0], pair[1], tweet)
        stripped_data.append(tweet)
    stripped = pd.Series(stripped_data)
    processed_data = list(map(re_tokens, stripped))
    lemmas = lemmatize(processed_data)
    return lemmas

def clean_stems(data):
    stripped_data = []
    subs = [(r'\{link\}', ''),
            (r'http\S+', ''),
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('(&amp)', ''),
            ('(&quot)', ''),
            ('(&nbsp)', ''),
            ('(&lt)', ''),
            ('(&gt)', ''),
            ('(RT\s)', '')
           ]
    for tweet in data:
        for pair in subs:
            tweet = re.sub(pair[0], pair[1], tweet)
        stripped_data.append(tweet)
    stripped = pd.Series(stripped_data)
    processed_data = list(map(re_tokens, stripped))
    stems = stemmatize(processed_data)
    return stems
