import string, re
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

def vocabulary(processed_data):
    vocabulary = set()
    for tweet in processed_data:
        vocabulary.update(tweet)
    return vocabulary

def tweet_tokens(tweet):
    stop_list = stopwords.words('english')
    stop_list += list(string.punctuation)
    stop_set = set(stop_list)
    tokens = nltk.word_tokenize(tweet)
    no_stopwords = [token.lower() for token in tokens if
                    token.lower() not in stop_set]
    return no_stopwords