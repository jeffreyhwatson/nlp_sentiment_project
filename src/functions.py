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

# re to find hashtags and keep (#)
r'\B#\w*[a-zA-Z]+\w*'

# re to find hashtags and remove (#)
"#([a-zA-Z0-9_]{1,50})"

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
