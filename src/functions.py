import string, re
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def tweet_tokens(tweet):
    stop_list = stopwords.words('english')
    stop_list += list(string.punctuation)
    stop_set = set(stop_list)
    tokens = nltk.word_tokenize(tweet)
    no_stopwords = [token.lower() for token in tokens if
                    token.lower() not in stop_set]
    return no_stopwords