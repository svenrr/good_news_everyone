import spacy
from multiprocessing import Pool

class Lemmatizer:
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
    def is_token_allowed(self, token):
        if (not token or not token.string.strip() or
            token.is_stop or token.is_punct):
            return False
        return True
        
    def lem_text(self, text):

        # Lower text
        text = text.lower()

        # Lemmatize text, remove stopwords
        doc = self.nlp(text)
        text = [token.lemma_ for token in doc if self.is_token_allowed(token)]

        return ' '.join(text).replace('"', '').replace("'", '')
    
    def lem_list(self, texts, cores=4):
        
        with Pool(processes=cores) as pool:
            result = pool.map(self.lem_text, texts)
            
        return result