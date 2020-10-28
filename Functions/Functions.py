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

# Sentiment Analysis with StanfordNLP
def Sentiment_StanfordNLP(text):
    from pycorenlp import StanfordCoreNLP
    import numpy as np
    nlpStanford = StanfordCoreNLP('http://localhost:9000')
    results = nlpStanford.annotate(text,properties={
        'annotators':'sentiment, ner, pos',
        'outputFormat': 'json',
        'timeout': 50000,
        })
    sentiment =[]
    for s in results["sentences"]:
        sentiment.append(s["sentiment"])
    new_sentiment = []
    for sent in sentiment:
        new_string = sent.replace("Negative", "-1").replace("Neutral", "0").replace("Positive","1").replace("Verynegative","-2").replace("Verypositive","2")
        new_sentiment.append(new_string)
    sentiment_mean = []
    for x in new_sentiment:
        sentiment_mean.append(int(x))
    return np.mean(sentiment_mean)