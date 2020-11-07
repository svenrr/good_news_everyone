import pandas as pd 
import numpy as np
import spacy
from collections import Counter
from string import punctuation
import math
import en_core_web_lg

nlp = en_core_web_lg.load()

###############################################################################################################

def read_article(file_name):
    return open(file_name,"r",encoding="utf-8").read()

###############################################################################################################

def get_summary(article_text, limit_percent=0.1):
    ''' Inputs: 
    * article_text = Text to summarize 
    * limit_percent = Percentages of sentences to keep (i.e. 0.1 for 10%)
    '''
    
    # Tokenize the input/article_text and return the important keywords 
    keywords = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    doc = nlp(article_text.lower()) 
    for token in doc: 
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue 
        if(token.pos_ in pos_tag):
            keywords.append(token.text) 
            
    # Normalize the weightage of keywords (0 to 1)
    freq_words = Counter(keywords) 
    max_freqs = Counter(keywords).most_common(1)[0][1] 
    for fw in freq_words:
        freq_words[fw] = (freq_words[fw]/max_freqs) 

    # Calculate the importance of each sentence (cumulative normalized value)
    sentence_strength={}
    for sent in doc.sents:
        for word in sent: 
            if word.text in freq_words.keys(): 
                if sent in sentence_strength.keys():
                    sentence_strength[sent] += freq_words[word.text]
                else:
                    sentence_strength[sent] = freq_words[word.text]
                    
    # Determine the number of sentences in the summarization (min. 3)
    if len(list(doc.sents)) > 30: 
        limit = math.ceil(len(list(doc.sents)) * limit_percent)
    elif len(list(doc.sents)) < 6: 
        return "Original Text has less than 5 sentences. No summary is needed."
    else: 
        limit = 3
    
    # Find the top sentences 
    summary = []
    sorted_sentences = sorted(sentence_strength.items(), key=lambda kv: kv[1], reverse=True) 
    
    counter = 0
    for i in range(len(sorted_sentences)): 
        summary.append(str(sorted_sentences[i][0]).capitalize()) 
        counter += 1
        if(counter >= limit):
            break 
    
    reading_time(article_text)
    print("# of sentences (pre):  {}".format(len(list(doc.sents))))
    print("# of sentences (post): {}".format(len(summary)))
    print("\n")
    get_tags(article_text), print("\n")
    print("---"*40)
    
    for i in range(len(summary)): 
        print("- ",summary[i],"\n")

    #return print(' '.join(summary))
    
############################################################################################################### 

def word_frequency(article_text):
    from spacy.lang.en.stop_words import STOP_WORDS
    stopwords = list(STOP_WORDS)
    
    doc = nlp(article_text.lower()) 
    
    word_freqs = {}
    for word in doc:
        if (word.text not in stopwords and word.text not in punctuation and word.text not in '“”'):
            if word.text not in word_freqs.keys():
                word_freqs[word.text] = 1
            else:
                word_freqs[word.text] += 1
                
    sort_orders = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
    print("Words that appear more than two times:")
    print("---"*40)
    for i in sort_orders:
        if i[1] > 2:
            print(i[0], i[1])

############################################################################################################### 

# Avg. reading time = 200-250 (https://irisreading.com/what-is-the-average-reading-speed/)
def reading_time(doc): # For the original article
    total_words = [token.text for token in nlp(doc)]
    time = len(total_words)/225
    print("Reading time for the entire article: {} mins (aprox)".format(round(time)))
    

############################################################################################################### 

def get_tags(article_text):
    from string import punctuation
    from spacy.lang.en.stop_words import STOP_WORDS
    stopwords = list(STOP_WORDS)
    
    # Just use nouns 
    pos_tag = ['NOUN']
            
    doc = nlp(article_text.lower()) 
    
    word_freqs = {}
    for word in doc:
        if (word.text not in stopwords and word.text not in punctuation and word.text not in '“”' and word.pos_ in pos_tag):
            if word.text not in word_freqs.keys():
                word_freqs[word.text] = 1
            else:
                word_freqs[word.text] += 1
                
    sort_orders = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
    
    for word in sort_orders[0:5]: 
        print("#",word[0],end=" ")

############################################################################################################### 
