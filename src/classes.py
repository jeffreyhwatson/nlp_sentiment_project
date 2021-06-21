import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_val_score)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             TfidfTransformer)
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             make_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
    
class Harness:
    
    def __init__(self, scorer, random_state=2021):
        self.scorer = scorer
        self.history = pd.DataFrame(columns=['Name', 'Accuracy (F1)', 'Notes'])

    def report(self, model, X, y, name, notes='', cv=5,):
        scores = cross_val_score(model, X, y, 
                                 scoring=self.scorer, cv=cv)
        frame = pd.DataFrame([[name, scores.mean(), notes]], columns=['Name', 'Accuracy (F1)', 'Notes'])
        self.history = self.history.append(frame)
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('Accuracy (F1)')
        self.print_error(name, scores.mean())
#         print(scores)
        return scores

    def print_error(self, name, Accuracy):
        print(f'{name} has an average F1 of {Accuracy}')



