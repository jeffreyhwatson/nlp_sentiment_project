import string, re
import pandas as pd
import numpy as np
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_score)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             TfidfTransformer)
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             make_scorer, plot_confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_sm_pipeline
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from src import classes as c

def pre_score(y_true, y_pred):
    "Precision scoring function for use in make_scorer."
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    return precision

# creating scorer object for pipelines
precision = make_scorer(pre_score)

def f_score(y_true, y_pred):
    "F1 scoring function for use in make_scorer."
    
    f1 = f1_score(y_true, y_pred)
    return f1

# creating scorer object for pipelines
f1 = make_scorer(f_score)

def framer(df, col, li):
    "Returns a data frame with selected columns."
    
    _list = [x for x in li if x not in col]
    column_list = df.columns
    cols = [x for x in column_list if x not in _list]
    return df[cols]

def Xy(df):
    """Returns a data frame and target series."""
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y

def splitter(X, y):
    """Returns a train/test split."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=2021,
                                                        stratify=y
                                                   )
    return  X_train, X_test, y_train, y_test

def confusion(model, X, y):
    "Returns a confusion matrix plot."
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap=plt.cm.Blues, 
                          display_labels=['Positive', 'Negative'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
#     plt.savefig('title',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()

def confusion_report(model, X, y):
    "Returns a confusion matrix plot."
    
    f1 = f1_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    precision = precision_score(y, model.predict(X))
    report = pd.DataFrame([[f1, recall, precision]],\
                          columns=['F1', 'Recall', 'Precision']) 
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap=plt.cm.Blues, 
                          display_labels=['Positive', 'Negative'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
#     plt.savefig('title',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()  
    
    return report 

    
def subsplit_test(model, X_train, y_train):
    """Returns train/test scores & a confusion matrix on subsplit test data."""
    
#     modeling = c.Harness(f1)
    Xs_train, Xs_test, ys_train, ys_test = splitter(X_train, y_train)
    model.fit(Xs_train, ys_train)
    train_score = f1_score(ys_train, model.predict(Xs_train))
    test_score = f1_score(ys_test, model.predict(Xs_test))
    confusion(model, Xs_train, ys_train)
    confusion(model, Xs_test, ys_test)
    recall_test = recall_score(ys_test, model.predict(Xs_test))
    precision_test = precision_score(ys_test, model.predict(Xs_test))
    report = pd.DataFrame([[train_score, test_score, recall_test, precision_test]],\
                          columns=['Train F1', 'Test F1', 'Test Recall', 'Test Precision'])
    return report    
    
def tokens(tweet):
    "Returns a list of tokens from a string"
    
    stop_list = stopwords.words('english')
    stop_list += ['sxsw', 'sxswi']
    stop_set = set(stop_list)
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tokens = tokenizer.tokenize(tweet)
    no_stopwords = [token.lower() for token in tokens if
                    token.lower() not in stop_set]
    return no_stopwords

def word_list(data):
    "Returns a flattened list of words from a list of word lists"
    
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
    subs = [
            ('\s[2]+(?![a-z])', ' '), #removes stray 2's from ipad 2
            (r'\{link\}', ''), #removes the string '{link}'
            (r'http\S+', ''), #removes urls
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''), #removes RTs
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''), #removes mentions
            ('(&amp)', ''), #removes '(&amp)' &
            ('(&quot)', ''), #removes '(&quot)' "
            ('(&nbsp)', ''), #removes '(&nbsp)' non-breaking spaces
            ('(&lt)', ''), #removes '(&lt)' less than
            ('(&gt)', ''), #removes '(&gt)' greater than
            ('(RT\s)', ''), #removes edgecase RT
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
            ('(RT\s)', ''),
            ('\s[2]+(?![a-z])', ''),
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
    subs = subs = [(r'\{link\}', ''),
            (r'http\S+', ''),
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('(&amp)', ''),
            ('(&quot)', ''),
            ('(&nbsp)', ''),
            ('(&lt)', ''),
            ('(&gt)', ''),
            ('(RT\s)', ''),
            ('\s[2]+(?![a-z])', ''),
           ]
    for pair in subs:
        tweet = re.sub(pair[0], pair[1], tweet)
    tweet = tokens(tweet)
    stems = ' '.join([ps.stem(word) for word in tweet])
    return list(stems.split())

def clean_corpus_stem(data):
    "Return a list of cleaned & stemmed words from a series of tweets."
    
    stripped_data = []
    subs = subs = [(r'\{link\}', ''),
            (r'http\S+', ''),
            ('RT\s@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('@[A-Za-z]+[A-Za-z0-9-_]+', ''),
            ('(&amp)', ''),
            ('(&quot)', ''),
            ('(&nbsp)', ''),
            ('(&lt)', ''),
            ('(&gt)', ''),
            ('(RT\s)', ''),
            ('\s[2]+(?![a-z])', '')
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
    "Returns a set of words from a series of tweets"
    
    return set([word for tweet in series for word in tweet.split()])

def word_frequencies(data, n):
    processed_data = list(map(tokens, data))
    word_li = word_list(processed_data)
    freqdist = FreqDist(word_li)
    word_count = sum(freqdist.values())
    top_n = freqdist.most_common(n)
    print("Word \t\t\tFrequency")
    print()
    for word in top_n:
        normalized_frequency = word[1]/word_count
        print(f'{word[0] : <10}\t\t{round(normalized_frequency, 4): <10}')

def ngrammer(series, n):
    n_grams = []
    for tweet in series:
        words = tweet.split()
        n_gram = list(ngrams(words, n=n))
        n_grams.append(n_gram)
    return n_grams