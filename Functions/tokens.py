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