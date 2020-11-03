import spacy
from multiprocessing import Pool

class Lemmatizer:
    
    def __init__(self):
        self.nlp = spacy.load('en')
        
    def lem_text(self, text):

        # Remove signs
        chars_to_remove = ':,!.\n-()/'
        for i in chars_to_remove:
            text = text.replace(i, '')

        # Lower text
        text = text.lower()

        # Lemmatize text, remove stopwords
        doc = self.nlp(text)
        text = [token.lemma_ for token in doc if not token.is_stop]

        return ' '.join(text).replace('"', '').replace("'", '')
    
    def lem_list(self, texts, cores=4):
        
        with Pool(processes=cores) as pool:
            result = pool.map(self.lem_text, texts)
            
        return result
