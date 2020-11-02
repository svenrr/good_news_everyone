def is_token_allowed(token):
    if (not token or not token.string.strip() or
        token.is_stop or token.is_punct):
        return False
    return True

def preprocess_token(token):
    return token.lemma_.strip().lower()

# Import this function to tokenize your text
def tokenize(text):
    import spacy
    from spacy.lang.en import English
    parser = English()
    unfiltered_tokens = parser(text)
    tokens = [preprocess_token(token) for token in unfiltered_tokens if is_token_allowed(token)]
    return tokens


######  Tokenization with Gensim #####

from gensim import utils
import gensim.parsing.preprocessing as gsp
from gensim.parsing.preprocessing import STOPWORDS
my_stop_words = STOPWORDS.union(set(['http', 'com', 'www']))
def preprocess(text):
    result = []
    for token in gsp.utils.simple_preprocess(text):
        if token not in my_stop_words:
            result.append(token)
    return ' '.join(result)

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
        s = preprocess(s)
    return s
