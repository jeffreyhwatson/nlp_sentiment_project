import string, re
import pandas as pd
import numpy as np
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def tokens(tweet):
    "Returns a list of tokens from a string"
    stop_list = stopwords.words('english')
    stop_set = set(stop_list)
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tokens = tokenizer.tokenize(tweet)
    no_stopwords = [token.lower() for token in tokens if
                    token.lower() not in stop_set]
    return no_stopwords

def word_list(data):
    "Returns a list of words from a list of word lists"
    vocab = [word for tweet in data for word in tweet]
    return vocab

def lemmatize(processed_data):
    "Returns a set of lemmatized words from a list of word lists"
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for doc in processed_data:
        lemms = ' '.join([lemmatizer.lemmatize(word) for word in doc])
        lemmas.append(lemms)
    return lemmas

def stemmatize(processed_data):
    "Returns a set of stemmed words from a list of word lists"
    ps = PorterStemmer()
    stemmed = []
    for doc in processed_data:
        stems = ' '.join([ps.stem(word) for word in doc])
        stemmed.append(stems)    
    return stemmed

def ht_extract(data):
    "Returns a list of hashtags from a series of tweets."
    hashlists = []
    for element in data:
        hashtag = re.findall(r'\B#\w*[a-zA-Z]+\w*', element)
        hashlists.append(hashtag)
    hashtags = [hashtag.lower() for h_list in hashlists for hashtag in h_list]
    return hashtags

def find_strings(data, expression):
    "Returns a list of words that match a given reg expression from a series."
    strings = []
    for tweet in data:
        string = re.findall(expression, tweet)
        if len(string) != 0:
            strings.append(string)
    return strings

def string_checker(data, string):
    "Return a string indicating if a given string in list/set of stings."
    if string in data:
        print('string is in data')
    else:
        print('string is not in data')

def clean_tweet_lem(tweet):
    "Return a list of cleaned & lemmatized strings from a tweet."
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    subs = [(r'\{link\}', ''), #removes the string '{link}'
            (r'http\S+', ''), #removes urls
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''), #removes RTs
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''), #removes mentions
            ('(&amp)', ''), #removes '(&amp)' &
            ('(&quot)', ''), #removes '(&quot)' "
            ('(&nbsp)', ''), #removes '(&nbsp)' non-breaking spaces
            ('(&lt)', ''), #removes '(&lt)' less than
            ('(&gt)', ''), #removes '(&gt)' greater than
            ('(RT\s)', '') #removes edgecase RT
           ]
             
    for pair in subs:
        tweet = re.sub(pair[0], pair[1], tweet)
    tweet = tokens(tweet)
    lemms = ' '.join([lemmatizer.lemmatize(word) for word in tweet])
    return list(lemms.split())

def clean_corpus_lem(data):
    "Return a list of cleaned & lemmatized words from a series of tweets."
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
    processed_data = list(map(tokens, stripped))
    lemmas = lemmatize(processed_data)
    return lemmas

def clean_tweet_stem(tweet):
    "Return a list of cleaned & stemmed strings from a tweet."
    ps = PorterStemmer()
    stems = []
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
    for pair in subs:
        tweet = re.sub(pair[0], pair[1], tweet)
    tweet = tokens(tweet)
    stems = ' '.join([ps.stem(word) for word in tweet])
    return list(stems.split())

def clean_corpus_stem(data):
    "Return a list of cleaned & stemmed words from a series of tweets."
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
    processed_data = list(map(tokens, stripped))
    stems = stemmatize(processed_data)
    return stems

def words(series):
    "Returns a list of words from a series of tweets"
    return [word for tweet in series for word in tweet.split()]

def vocabulary(series):
    "Returns a list of words from a series of tweets"
    return set([word for tweet in series for word in tweet.split()])

def top_word_list(data, n, print_list=False, return_list=False):
    """Plots a FreqDist plot, 
       optionally prints and/or returns the n most common words in a corpus."""
    processed_data = list(map(tokens, data))
    word_li = word_list(processed_data)
    freqdist = FreqDist(word_li)
    most_common = freqdist.most_common(n)
    top_word_list = [tup[0] for tup in most_common]
    plt.figure(figsize=(15,7))
    freqdist.plot(n)
    # plt.savefig('title',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    if print_list == True:
        print(top_word_list)
    if return_list == True:
        return top_word_list
    
    