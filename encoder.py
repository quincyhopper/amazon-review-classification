"""Classes for:
    - BagOfWords vectorisation
    - GloVe embedding vectorisation
    - TF-IDF weighting
"""

import numpy as np
import pickle
from itertools import chain
from tqdm import tqdm
from collections import Counter

class BagOfWords:
    """Class for creating one hot vectors.

    Attributes:
        tfidf (TFIDF): Optional instance of TFIDF class used for weighting vectors by TF-IDF score.
        vocab (dict): Dictionary of unique tokens in corpus. Keys are index, values are token.
        vocab_size (int): Number of unique tokens in corpus.
    """
    def __init__(self, tfidf:"TFIDF"=None):
        """Initialise BagOfWords class.
        
        Args: 
            tfidf (TFIDF): The TFIDF instance used for weighting vectors.
        """
        self.tfidf = tfidf
        self.vocab = None
        self.vocab_size = 0.0

    def fit(self, tokenised_reviews:list[list]) -> None:
        """Create dictionary of unique tokens in corpus.
        
        Args:
            tokenised_reviews: NumPy object array where each element is a list of string tokens.
        """
        
        # Flatten to get unique tokens in the corpus
        flat_types = sorted(set(chain.from_iterable(tokenised_reviews)))

        # Give an index to each word
        self.vocab = {word: i for i, word in enumerate(flat_types)}
        self.vocab_size = len(self.vocab)

    def transform(self, tokenised_reviews:np.ndarray) -> np.ndarray:
        """Convert tokenised reviews to one-hot encoded matrix. 
        
        If class is passed a tfidf instance, each review's vector is weighted by it's TF-IDF score.
        
        Args:
            tokenised_reviews (np.ndarray): NumPy object-array where each element is a list of string tokens.

        Returns:
            One-hot encoded (or TF-IDF weighted) matrix of shape (n_reviews, vocab_size).
        
        Raises:
            ValueError: If fit() is called before transform(). 
        """
        if self.vocab is None:
            raise ValueError("Must call fit() before transform()")
        
        # Initialise zero matrix of shape (n_reviews, vocab_size)
        n_reviews = len(tokenised_reviews)
        X = np.zeros((n_reviews, self.vocab_size), dtype=np.float32)

        # One hot encode each review into a matrix
        for row_idx, review in enumerate(tqdm(tokenised_reviews, desc="Vectorising")):
            unique_words = set(review) # Only look at unique words to save time

            for word in unique_words:
                if word in self.vocab:
                    col_idx = self.vocab[word] # Get column index

                    if self.tfidf:
                        # Weight by TF-IDF score
                        X[row_idx, col_idx] = self.tfidf.get_weight(word, review)
                    else:
                        # Otherwise, set value to 1
                        X[row_idx, col_idx] = 1

        return X
    
    def fit_transform(self, reviews:list) -> np.ndarray:
        """fit() and transform() in one go."""
        self.fit(reviews)
        return self.transform(reviews)
    
    def save(self, filename:str) -> None:
        """Save vocab dictionary to .pkl file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

    
class GloveVectoriser():
    """Class for creating matrix of GloVe embeddings. 
    
    Attributes:
        tfidf (TFIDF): Optional instance of TFIDF class used for weighting vectors by TF-IDF score.
        embeddings (dict): Dictionary of words and their respective GloVe embeddings.
        dim: Dimensions of GloVe embeddings.
    """
    def __init__(self, version:str='200d', tfidf=None) -> None:
        """Initialise GloveVectoriser and load pre-trained embeddings.
        
        Args:
            version (str): GloVe embedding to load. Supports '100d' and '200d'. Default is '200d'.
            tfidf (TFIDF): Optional instance of TFIDF class used for weighting vectors by TF-IDF score. 

        Raises:
            ValueError: If version is not '100d' or '200d'.
        """
        if version not in ['100d', '200d']:
            raise ValueError("GloVe version must be '100d' or '200d'.")
        
        self.tfidf = tfidf
        self.embeddings = {}

        if version == '100d':
            self.dim = 100
            filepath = 'data/glove.100d.txt'
        elif version == '200d':
            self.dim = 200
            filepath = 'data/glove.200d.txt'

        with open(filepath, 'r') as f:
            for line in tqdm(f, desc="Loading GloVe embeddings"):
                x = line.split()
                self.embeddings[x[0]] = np.asarray(x[1:], dtype='float32')

    def fit(self) -> None:
        """Empty function for consistency."""
        pass
    
    def transform(self, tokenised_reviews:np.ndarray) -> np.ndarray:
        """Convert tokenised_reviews to average of review's token embeddings. 
        
        If class is passed a tfidf instance, each review's vector is a TF-IDF weighted average of all tokens in review.
        
        Args:
            tokenised_reviews (np.ndarray): NumPy object-array where each element is a list of string tokens.

        Returns:
            Matrix of shape (n_samples, GloVe_dimensions) of average (or TF-IDF weighted average) GloVe embeddings for each review.
        """

        # Initialise matrix of zeros
        n_reviews = len(tokenised_reviews)
        X = np.zeros((n_reviews, self.dim), dtype=np.float32)

        # Loop over each review
        for row_idx, review in enumerate(tqdm(tokenised_reviews, desc='Vectorising')):
            
            # Initialise review vector of zeros
            v = np.zeros(self.dim)

            # Initialise count for summing the weights
            weight_sum = 0

            # Loop over every token in the review
            for t in review:
                if t in self.embeddings:

                    # Weight is either 1 or TF-IDF score
                    if self.tfidf:
                        weight = self.tfidf.get_weight(t, review)
                    else:
                        weight = 1.0

                    # Add weight to sum of weights
                    weight_sum += weight

                    # Add the token's weighted embedding to review vector
                    v += self.embeddings[t] * weight

            # weight_sum is 0 if none of the tokens have corresponding embeddings. This if statement avoids divisoin by zero
            if weight_sum > 0:

                # Calculate average embedding for review
                v_avg = v / weight_sum

                # Normalise vector 
                norm = np.linalg.norm(v_avg)
                if norm > 0:
                    X[row_idx] = v_avg / norm 
                else:
                    X[row_idx] = v_avg

        return X

    def fit_transform(self, tokenised_reviews:np.ndarray) -> np.ndarray:
        """fit() and transform() in one go.
        
        Args:
            tokenised_reviews (np.ndarray): NumPy object-array where each element is a list of string tokens.

        Returns:
            NumPy array of shape (n_samples, GloVe_dimensions).
        """
        self.fit()
        return self.transform(tokenised_reviews)

class TFIDF():
    """Class for handling TF-IDF scores.
    
    Attributes:
        idf(dict): Dictionary of idf score for each unique token in corpus.
    """
    def __init__(self):
        self.idf = {}

    def learn_idf(self, tokenised_reviews:np.ndarray) -> None:
        """Calculate IDF score for every token in corpus"""

        n_reviews = len(tokenised_reviews)
        token_counts = Counter()

        # Count how many times each type occurs in review
        for review in tokenised_reviews:
            types = set(review)
            token_counts.update(types)

        # Compute idf score for each token
        self.idf = {
            token: np.log((n_reviews) / (count))
            for token, count in token_counts.items()
        }

    def get_weight(self, token:str, tokenised_review:np.ndarray) -> np.float32:
        """Calculate TF-IDF score for given token in a review
        
        Args:
            token: The token to calculate TF-IDF score for.
            tokenised_reviews: Numpy object containg lists of tokenised reviews, where each review is a list of tokens.

        Returns:
            0.0 if token has no idf score, else TF-IDF score for token.
        """
        if token not in self.idf:
            return 0.0
        
        # Handle both NumPy arrays and lists
        if isinstance(tokenised_review, np.ndarray):
            count = np.sum(tokenised_review == token)
        else:
            count = tokenised_review.count(token)

        tf = count / len(tokenised_review)
        return tf * self.idf[token]
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.idf, f)