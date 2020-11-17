import numpy as np
from gensim.models import KeyedVectors
class EmbeddingVectorizer(object):
    def __init__(self, word_vectors='standard.word_vectors', method='mean', vocab_count=False):
        self.word_vectors = KeyedVectors.load(word_vectors, mmap='r')
        self.dim = self.word_vectors.vector_size
        self.method = method
        self.vocab_count = vocab_count

    def transform(self, X):
        # Turn string to tokens
        X = [s.split(' ') for s in X]
        
        if self.method == 'mean':
            return np.array([
                np.concatenate((
                np.mean(
                [self.word_vectors[w] for w in words if w in self.word_vectors.vocab]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'sum':
            return np.array([
                np.concatenate((
                np.sum(
                [self.word_vectors[w] for w in words if w in self.word_vectors.vocab]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'full':
            return np.array([
                np.concatenate((
                np.array([self.word_vectors[w] if w in words else np.zeros(self.dim) for w in self.word_vectors.vocab]).flatten(),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        
        
class ImportanceVectorizer(object):
    def __init__(self, word_vectors, method='mean', vocab_count=False):
        self.word_vectors = word_vectors
        self.dim = self.word_vectors.vector_size
        self.method = method
        self.vocab_count = vocab_count

    def transform(self, X):
        # Turn string to tokens
        X = [s.split(' ') for s in X]
        
        if self.method == 'mean':
            return np.array([
                np.concatenate((
                np.mean(
                [self.word_vectors[w] for w in words if w in self.word_vectors.vocab]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'sum':
            return np.array([
                np.concatenate((
                np.sum(
                [self.word_vectors[w] for w in words if w in self.word_vectors.vocab]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'full':
            return np.array([
                np.concatenate((
                np.array([self.word_vectors[w] if w in words else np.zeros(self.dim) for w in self.word_vectors.vocab]).flatten(),
                np.array([words.count(w) for w in self.word_vectors.vocab if self.vocab_count])))
                for words in X
            ])
        
        
class GloveVectorizer(object):
    def __init__(self, word_vectors, method='mean', vocab_count=False):
        self.word_vectors = word_vectors
        self.dim = self.word_vectors[list(self.word_vectors.keys())[0]].shape[0]
        self.method = method
        self.vocab_count = vocab_count

    def transform(self, X):
        # Turn string to tokens
        X = [s.split(' ') for s in X]
        
        if self.method == 'mean':
            return np.array([
                np.concatenate((
                np.mean(
                [self.word_vectors[w] for w in words if w in self.word_vectors.keys()]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.keys() if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'sum':
            return np.array([
                np.concatenate((
                np.sum(
                [self.word_vectors[w] for w in words if w in self.word_vectors.keys()]
                    or [np.zeros(self.dim)], axis=0),
                np.array([words.count(w) for w in self.word_vectors.keys() if self.vocab_count])))
                for words in X
            ])
        
        elif self.method == 'full':
            return np.array([
                np.concatenate((
                np.array([self.word_vectors[w] if w in words else np.zeros(self.dim) for w in self.word_vectors.keys()]).flatten(),
                np.array([words.count(w) for w in self.word_vectors.keys() if self.vocab_count])))
                for words in X
            ])