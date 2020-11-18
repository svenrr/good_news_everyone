import pickle
import os
from text_lemmatization import Lemmatizer

class Categorizer():
    def __init__(self):

        # get total path, in case function is called from somewhere else
        total_path = os.path.dirname(os.path.realpath(__file__)) + '/'

        self.tfidf = pickle.load(open(total_path + 'tfidf_categorizer.pkl', 'rb'))
        self.lemmatizer = Lemmatizer()
        self.svc = pickle.load(open(total_path + 'svc_categorizer.pkl', 'rb'))
    def preprocess(self, X):
        
        # Check if X is string, turn to list
        if type(X) == str:
            X = [X]
                    
        # Lemmatization
        X_lem = [self.lemmatizer.lem_text(x) for x in X]
                
        # Tfidf vectorization
        X_tfidf = self.tfidf.transform(X_lem)
        
        return X_tfidf
    
    def pred(self, X):
        
        # preprocess
        X_tfidf = self.preprocess(X)
        
        # return categories
        return self.svc.predict(X_tfidf)
