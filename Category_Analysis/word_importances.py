from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def word_importances(df):

    text = df.text_lem.values
    # Number of times the words have to occur at least
    min_df=10

    # ngram_range only use 1 word, stop_words=english remove common words(eg: a, an, the)
    # CountVectorizer ignores difference between lower and upper case and puncuation
    vect = CountVectorizer(ngram_range=(1,2), stop_words='english', min_df=min_df)

    # CountVectorizer converts list of strings to matrix with: rows = observation, columns = terms in text, values=count/document
    X = vect.fit_transform(text)
    words = vect.get_feature_names()


    # Turn result to dummies, so that columns are in correct order
    d = {}
    count=0
    for i in df.category.unique():
        d[i] = count
        count += 1
    df['category_num'] = [d[i] for i in df.category]

    y = df.category_num

    # Remove alpha to prevent bias
    clf = MultinomialNB(alpha=1.e-10)
    clf.fit(X,y)

    likelihood_df_raw = pd.DataFrame(clf.feature_log_prob_.transpose(),columns=df.category.unique(), index=words)


    likelihood_df = {}
    for category in df.category.unique():
        likelihood_df[category] = np.exp(likelihood_df_raw[category]) - np.sum(np.exp(likelihood_df_raw.drop(category, axis=1)), axis=1)
        #likelihood_df[category] = np.exp(likelihood_df_raw[category]) - np.exp(np.sum(likelihood_df_raw.drop(category, axis=1), axis=1))

        # Sort likelihood
        likelihood_df[category] = likelihood_df[category].sort_values(ascending=False)
        
    return pd.DataFrame(likelihood_df)