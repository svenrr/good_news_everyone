"""
                SIMULATION

Run a simulation, where words attract each other by the number of articles they are in together,
while repulse each other by the number of articles they are not in together. The more occurences
a word has, the higher its attraction and repulsion is.
"""

### Settings
ITERATIONS=10000
VOCAB_SIZE=30 # Number of vocabularies per category
OUTPUT_FOLDER='simulation_results' # Folder to save results
#ATT_LIMIT = 0.5
ATT_FORCE = 50 # Attractive force between words
REP_FORCE = 150 # Repulsive force between words
SAMPLES_PER_LABEL=1000#100 # Articles used per category
MIN_ATT=50 # Minimal limit word count is normalized to
MAX_ATT=60 # Maximal limit word count is normalized to
VECTORIZED=True # Use vectorized version, much faster
MAX_STEP = 3 # Maximal step length a word is allowed to take in one iteration

#### Imports
import pandas as pd
import numpy as np
from time import time
import pickle
# Add functions path
import sys
sys.path.append('../../Functions')
from datasets import load_stratified_dataset

"""
            Naive Bayes for word importances 
"""
def word_importances():
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

    return likelihood_df, words, X, likelihood


"""
            Choose Vocabulary
"""
def get_vocab():
    vocab_size = VOCAB_SIZE

    # Get vocabulary
    vocabulary = set()
    for category in likelihood.keys():
        vocabulary.update(likelihood[category].index[:vocab_size])
    vocabulary = list(vocabulary)

    return vocabulary
    
"""
        Calculate word count

Get the word counts, normalize them between two values, take this as attractive / repulsive force per word
"""
def get_word_count():

    # Total count of words
    count_total = dict(zip(words, X.toarray().sum(axis=0)))

    # Normalize word counts between two values
    min_att, max_att = MIN_ATT, MAX_ATT
            
    l = []
    for word in words:
        l.append(count_total[word])
    min_count = min(l)
    max_count = max(l)

    force_per_word = [(count_total[word]-min_count)/(max_count-min_count)*(max_att-min_att) + min_att for word in words]
    force_per_word = np.array(force_per_word)

    return force_per_word, count_total

"""
        Calculate the Forces between word pairs
"""
def get_force():

    start_time = time()
    num_articles = Distribution.shape[0]
    if VECTORIZED:
        Forces_att = np.empty((words.shape[0], words.shape[0]))
        Forces_rep = np.empty((words.shape[0], words.shape[0]))
    else:
        Matches = np.empty((words.shape[0], words.shape[0]))
    for w1 in range(words.shape[0]):
        for w2 in range(words.shape[0]):
            dist1 = Distribution[:,w1]
            dist2 = Distribution[:,w2]
            if VECTORIZED:
                # Number of articles both words share
                matches = ((dist1 > 0) & (dist2 > 0)).sum()
                #Forces_no_dist[w1,w2] = force_per_word[w1] * force_per_word[w2] * (ATT_LIMIT - matches/num_articles)
                Forces_att[w1,w2] = force_per_word[w1] * force_per_word[w2] * matches/num_articles
                Forces_rep[w1,w2] = force_per_word[w1] * force_per_word[w2] * (num_articles - matches)/num_articles

            else:
                Matches[w1,w2] = ((dist1 > 0) & (dist2 > 0)).sum()
    print('Matches calculated in {:.2f}m'.format((time()-start_time)/60))            

    # Convert Forces for each word pair to vector[[1-2, 2-3, 3-4, ...], [1-3, 2-4,...], ...]
    # Get the diagonals under the main diagonal
    if VECTORIZED:
        start_time = time()
        Forces_att = np.array([np.diag(np.roll(Forces_att, -x, axis=0)).reshape(Forces_att.shape[0],1) for x in range(0, words.shape[0])])
        Forces_rep = np.array([np.diag(np.roll(Forces_rep, -x, axis=0)).reshape(Forces_rep.shape[0],1) for x in range(0, words.shape[0])])
        print('Matches / Forces transformed in {:.2f}m'.format((time()-start_time)/60))

    if VECTORIZED:
        return Forces_att, Forces_rep
    else:
        return Matches

"""
        Euclidian Distance
"""
def calc_dist(word1, word2):
    p1 = Map[word1]
    p2 = Map[word2]
    
    vec = p1-p2
    dist = np.linalg.norm(vec)
    #dist = np.square(np.square(p1-p2).sum())

    #print(vec)
    #exit()
    
    return dist, vec/dist


"""
        Euclidian Distance Vectorized
"""
def calc_dist_vectorized(Map1, Map2):
    
    Vec = Map1 - Map2
    Dist = np.linalg.norm(Vec, axis=1).reshape(Vec.shape[0],1)
    #print(Vec)
    #exit()
    return Dist, Vec/Dist

"""
        Calculate Interaction between two words

Words attract each other for each equal article
"""
def interaction(word1, word2):    
    
    # Get number of articles the words share
    matches = Forces[word1, word2]

    # Get distance between words
    dist, vec = calc_dist(word1, word2)
    
    # Calculate force (include small value because of start point)
    num_articles = Distribution.shape[0]
    f1 = force_per_word[word1]
    f2 = force_per_word[word2]

    #force = f1*f2*(ATT_LIMIT-matches/num_articles) / (dist + 1E-5)**2
    att_force = ATT_FORCE * f1*f2 * matches / (dist + 1E-5) / num_articles
    rep_force = REP_FORCE * f1*f2 * (num_articles - matches) / (dist + 1E-5)**2 / num_articles
    force = (att_force - rep_force)

    # Add force to move
    Move[word1, :] += vec*force
    Move[word2, :] -= vec*force

"""
        Calculate word Interaction vectorized
"""
def interaction_vectorized(roll_x, Move):

    # Shift map, so every word is in same place as next word and so on
    Map2 = np.roll(Map, -roll_x, axis=0)

    Dist, Vec = calc_dist_vectorized(Map, Map2)

    Att_force = ATT_FORCE * (Forces_att[roll_x] / (Dist + 1E-5))
    Rep_force = REP_FORCE * (Forces_rep[roll_x] / (Dist + 1E-5)**2)
    Force = (Att_force - Rep_force)
    #Force = Forces[roll_x] / (Dist + 1E-5)**2

    # Add force to move
    Move -= Vec * Force

    return Move

"""
        Clip Movement if too large
"""
def clip():
    for i in range(Move.shape[0]):
        dist = np.linalg.norm(Move[i,:])
        if abs(dist) > MAX_STEP:
            Move[i,:] = Move[i,:]/dist * MAX_STEP


"""
            Save stuff
"""
def save_content(it=''):

    pickle.dump(Map_History, open(OUTPUT_FOLDER + "/Map_History" + it + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump(words, open(OUTPUT_FOLDER + "/words" + it + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump(likelihood_df, open(OUTPUT_FOLDER + "/likelihood_df" + it + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump(count_total, open(OUTPUT_FOLDER + "/count_total" + it + ".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)




"""
            Initialize Map
"""

#### Load dataset
df = load_stratified_dataset(
    path='../../Datasets/dataset_categories/dataset_categories_train.csv',
    labels='category', samples_per_label=SAMPLES_PER_LABEL, random_seed=42)

# Get word importances
start = time()
likelihood_df, words, X, likelihood = word_importances()
print('Created word importances in {:.2f}m'.format((time()-start)/60))

# Get vocabulary
vocabulary = get_vocab()

# Get words, Distribution of words over articles
words = np.array(words)
Distribution = X.toarray()

# Only select vocabulary
mask = [True if word in vocabulary else False for word in words]
Distribution = Distribution[:,mask]
words = words[mask]

# Initialize Map randomly, create Map_History
np.random.seed(42)
Map = np.random.randn(words.shape[0], 2)*5
Map_History = []
Map_History.append(Map.copy())

# Calculate normalized word count
force_per_word, count_total = get_word_count()

# Calculate forces between words without distance
if VECTORIZED:
    Forces_att, Forces_rep = get_force()
else:
    Forces = get_force()


"""
            Run Simulation
"""
full_start_time = time()
# Iterate x times
for it in range(1, ITERATIONS+1):
    start_time = time()
    # Set Move to zero
    Move = np.zeros([words.shape[0], 2])

    if VECTORIZED:
        # Run all words over all words
        for roll_x in range(1,words.shape[0]):
            Move = interaction_vectorized(roll_x, Move)
    else:
        # Run all words over all words
        for word1 in range(words.shape[0]):
            for word2 in range(word1+1, words.shape[0]):
                interaction(word1, word2)

    # Clip Movement
    clip()
    # Update Map
    Map += Move
    Map_History.append(Map.copy())

    # Save intermediates
    if it % 1000 == 0:
        save_content("_" + str(it))

    print('Iteration {} finished in {:.2f}m'.format(it+1, (time()-start_time)/60), end='\r')

print('Finished {} iterations in {:.2f}m                  '.format(ITERATIONS, (time() - full_start_time)/60))
#print(Map.sum())

save_content()