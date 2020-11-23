"""
                SIMULATION

Run a simulation, where words attract each other by the number of articles they are in together,
while repulse each other by the number of articles they are not in together. The more occurences
a word has, the higher its attraction and repulsion is.
"""

ITERATIONS=10000
VOCAB_SIZE=30#30
OUTPUT_FOLDER='simulation_results'
ATT_FORCE=50
REP_FORCE=150
SAMPLES_PER_LABEL=300#100
MIN_ATT=50
MAX_ATT=60
#LOAD_MAP=True


#### Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
# Add functions path
import sys
sys.path.append('../../Functions')
from datasets import load_stratified_dataset
from sklearn.cluster import KMeans

#### Load dataset
df = load_stratified_dataset(path='../../Datasets/dataset_categories/dataset_categories_train.csv', labels='category', samples_per_label=SAMPLES_PER_LABEL)


"""
            Naive Bayes for word importances 
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

text = df.text_lem.values
# Number of times the words have to occur at least
min_df=10

# ngram_range only use 1 word, stop_words=english remove common words(eg: a, an, the)
# CountVectorizer ignores difference between lower and upper case and puncuation
vect = CountVectorizer(ngram_range=(1,1), stop_words='english', min_df=min_df)

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

likelihood = {}
for category in df.category.unique():
    likelihood[category] = np.exp(likelihood_df_raw[category]) - np.sum(np.exp(likelihood_df_raw.drop(category, axis=1)), axis=1)
    
    # Sort likelihood
    likelihood[category] = likelihood[category].sort_values(ascending=False)
    
likelihood_df = pd.DataFrame(likelihood)

"""
            Most used words
"""

#count_total = pd.DataFrame({'word': words, 'count': X.toarray().sum(axis=0)})
#count_total.sort_values('count', ascending=False).head(5)

count_total = dict(zip(words, X.toarray().sum(axis=0)))

"""
            Word count per category
"""

# Create list containing empty list for each category
l = [[] for i in df.category.unique()]
ct = 1
total = y.shape[0]
# Append vectors depending on their category
for category, count in zip(y, X.toarray()):
    print('Get vectors for each category: {:.2f}%'.format(ct/total*100), end='\r')
    ct+=1
    l[category].append(count)

l2 = []
count=1
# Get the sum of counts for each category
for cat in l:
    print('Count words for each category: {:.2f}%'.format(count/len(l)*100), end='\r')
    count+=1
    cat = np.array(cat)
    tmp = cat.sum(axis=0)
    l2.append(tmp)
    
l2 = np.array(l2)

# Transform counts in Dataframe
word_count = pd.DataFrame(l2.transpose(), columns=df.category.unique(), index=words)

"""
            Choose Vocabulary
"""

if True:
    vocab_size = VOCAB_SIZE

    # Get vocabulary
    vocabulary = set()
    for category in likelihood.keys():
        vocabulary.update(likelihood[category].index[:vocab_size])
    vocabulary = list(vocabulary)
else:
    vocabulary = words
    

"""
            Create Force Field
"""

# X(articles, word_count); y(category); words(words)
X.toarray().shape, y.shape, np.array(words).shape

# Get words, Distribution of words over articles
words = np.array(words)
Distribution = X.toarray()

# Only select vocabulary
mask = [True if word in vocabulary else False for word in words]
Distribution = Distribution[:,mask]
words = words[mask]

# Map and Map history over iterations
Map_History = []

# Initialize Map randomly
np.random_seed = 42
Map = np.random.randn(words.shape[0], 2)*5
Map_History.append(Map.copy())

# Calculate attraction/repulsion
# The more often the word appears, the higher the attraction/repulsion
min_att, max_att = MIN_ATT, MAX_ATT
        
l = []
for word in words:
    l.append(count_total[word])
min_count = min(l)
max_count = max(l)

force_per_word = [(count_total[word]-min_count)/(max_count-min_count)*(max_att-min_att) + min_att for word in words]

# Calculate euclidian distance between objects
def calc_dist_vec(word1, word2):
    p1 = Map[word1]
    p2 = Map[word2]
    
    vec = p1-p2
    dist = np.linalg.norm(vec)
    #dist = np.square(np.square(p1-p2).sum())
    
    return dist, vec/dist

# Calculate Interaction between two words
# Words attract each other for each equal article
def interaction(word1, word2):    
    # Get distribution in articles of both words
    dist1 = Distribution[:, word1]
    dist2 = Distribution[:, word2]

    # Get number of articles the words share
    matches = ((dist1 > 0) & (dist2 > 0)).sum()

    # Get distance between words
    dist, vec = calc_dist_vec(word1, word2)
    
    # Calculate force (include small value because of start point)
    num_articles = Distribution.shape[0]
    f1 = force_per_word[word1]
    f2 = force_per_word[word2]
    att_force = ATT_FORCE * f1*f2 * matches / (dist + 0.01) / num_articles
    rep_force = REP_FORCE * f1*f2 * (num_articles - matches) / (dist + 0.01)**2 / num_articles
    #print(opp_force)

    # Add force to move
    Move[word1, :] -= vec*(att_force - rep_force)
    Move[word2, :] += vec*(att_force - rep_force)

# Clip Movement if too large
def clip():
    for i in range(Move.shape[0]):
        dist = np.linalg.norm(Move[i,:])
        if abs(dist) > 3:
            Move[i,:] = Move[i,:]/dist * 3


"""
            Run Simulation
"""

# Iterate x times
for it in range(ITERATIONS):
    start_time = time()
    # Set Move to zero
    Move = np.zeros([words.shape[0], 2])
    # Run all words over all words

    for word1 in range(words.shape[0]):
        for word2 in range(word1+1, words.shape[0]):

            # Calculate interaction
            interaction(word1, word2)

    # Clip Movement
    clip()
    # Update Map
    Map += Move
    Map_History.append(Map.copy())

    if (it+1)%1000 == 0:
        pickle.dump(Map_History, open(OUTPUT_FOLDER + "/Map_History_" + str(it+1) + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(words, open(OUTPUT_FOLDER + "/words_" + str(it+1) + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(likelihood_df, open(OUTPUT_FOLDER + "/likelihood_df_" + str(it+1) + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(count_total, open(OUTPUT_FOLDER + "/count_total_" + str(it+1) + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print('Iteration {} finished in {:.2f}m'.format(it+1, (time()-start_time)/60), end='\r')


"""
            Save stuff
"""

pickle.dump(Map_History, open(OUTPUT_FOLDER + "/Map_History.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(words, open(OUTPUT_FOLDER + "/words.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(likelihood_df, open(OUTPUT_FOLDER + "/likelihood_df.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

pickle.dump(count_total, open(OUTPUT_FOLDER + "/count_total.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)