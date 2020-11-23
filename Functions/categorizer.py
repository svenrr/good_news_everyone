import pickle
import os
from text_lemmatization import Lemmatizer
import numpy as np

class Categorizer():
    """
    *** Function to Categorize articles ***
    use pred() function to predict categories of one or multiple articles
    if show_words = True it will also return the 5 most important words 
    relevant for the decision
    """
    def __init__(self, show_words=False):

        # get total path, in case function is called from somewhere else
        total_path = os.path.dirname(os.path.realpath(__file__)) + '/'

        self.tfidf = pickle.load(open(total_path + 'tfidf_categorizer.pkl', 'rb'))
        self.lemmatizer = Lemmatizer()
        self.clf = pickle.load(open(total_path + 'clf_categorizer.pkl', 'rb'))
        self.show_words = show_words

    def preprocess(self, X):
        
        # Check if X is string, turn to list
        if type(X) == str:
            X = [X]
                    
        # Lemmatization
        X_lem = [self.lemmatizer.lem_text(x) for x in X]
                
        # Tfidf vectorization
        X_tfidf = self.tfidf.transform(X)
        
        return X_tfidf, X_lem
    
    def pred(self, X):
        
        # preprocess
        X_tfidf, X_lem = self.preprocess(X)

        # Get most important words
        if self.show_words:
            labels = self.clf.predict(X_tfidf)
            return labels, [self.get_imp_feat(label, x) for label, x in zip(labels, X_lem)]
        
        # return categories
        return self.clf.predict(X_tfidf)

    def get_imp_feat(self, label, X):
        # Get words that the features represent
        feature_names = self.tfidf.get_feature_names()

        # For given label
        idx = np.where(self.clf.classes_ == label)[0][0]

        # Sort coefficients, get their arguments
        sort_by_importance = np.argsort(self.clf.coef_[idx])[::-1]
        
        # Find first 5 most important words in X
        most_imp = []
        count=0
        for arg in sort_by_importance:
            count+=1
            word = feature_names[arg]
            # Add spaces else string can also be part of word
            if ' ' + word + ' ' in X:
                most_imp.append(word)
            if len(most_imp) > 4:
                break
        
        return most_imp