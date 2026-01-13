import numpy as np
import string
import re
from collections import Counter
from tqdm import tqdm

# Removed 'very', 'not', 'on', 'off'
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 
    'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don', 
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 
    'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 
    'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'now', 'of', 
    'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 
    'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 
    'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 
    'up', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 
    'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 
    'yourselves', 
}

def tokenise(reviews:list[str]) -> np.ndarray:
    """Tokenise reviews by splitting on whitespace or anything that is not alphabet or numbers.
    
    Args:
        reviews: List containing strings of reviews.
        
    Returns:
        NumPy object array where each element is a list of string tokens.
    """
    
    # Split on (1) any number of whitespace OR (2) any number of digits OR (3) anything that is not alphabet or digits or whitespace.
    delim = r"(\s+|\d+|[^a-zA-Z0-9\s])"

    tokenised_reviews = []
    for review in tqdm(reviews, desc="Tokenising"):
        
        # Split review
        tokens = re.split(delim, review)

        # Remove whitespace and empty tokens
        tokens = [t for t in tokens if t and not t.isspace()]
        tokenised_reviews.append(tokens)

    return np.array(tokenised_reviews, dtype='object')            

class Cleaner():
    """Class for cleaning tokenised reviews
    
    Attributes:
        allowed_tokens (set): Set of tokens that meet count threshold.
    """
    def __init__(self, lowercase=True, remove_punct=True, remove_digits = True, use_num=True, remove_stop_words=True, count_threshold=None):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.use_num = use_num
        self.remove_stop_words = remove_stop_words
        self.threshold = count_threshold
        self.allowed_tokens = None

    def clean_token(self, t:str) -> str|None:
        """Clean token according to class initialisation settings.
        
        Args:
            t (str): Token to be cleaned.
            
        Returns:
            Cleaned token, or None if the token is discarded by cleaning.
        """

        # Remove any whitespace
        t = t.strip()

        # Lowercase
        if self.lowercase:
            t = t.lower()
        
        # Check if token is punctuation
        if self.remove_punct and t in string.punctuation: 
            return None
        
        # Check if token is digit
        if self.remove_digits and t.isdigit():
            if self.use_num:
                return "num"
            else:
                return None
        
        # Check if token is a stop word
        if self.remove_stop_words and t in STOP_WORDS: 
            return None 
        
        return t

    def fit(self, tokenised_reviews:list[list]) -> None:
        """Count tokens in reviews and make set of tokens that meet count threshold.

        If `self.count_threshold` is None, does nothing.
        
        Args:
            tokenised_reviews (np.ndarray): Numpy object containg lists of tokenised reviews, where each review is a list of tokens.
        """
        if self.threshold is None:
            return

        token_counts = Counter()

        for review in tqdm(tokenised_reviews, desc='Learning thresholds'):
            for t in review:
                cleaned_token = self.clean_token(t)
                if cleaned_token:
                    token_counts.update([cleaned_token])

        # Create set of tokens that meet threshold
        self.allowed_tokens = set(t for t, c in token_counts.items() if c >= self.threshold)

    def transform(self, tokenised_reviews:np.ndarray[list]) -> np.ndarray:
        """Clean tokens in review.
        
        Args:
            tokenised_reviews (np.ndarray): Numpy object containg lists of tokenised reviews, where each review is a list of tokens.

        Returns:
            NumPy object array where each element is a list of string tokens that passed cleaning.
        """
        cleaned_corpus = []

        # Loop over each review 
        for review in tqdm(tokenised_reviews, desc="Cleaning tokens"):
            cleaned_review = []

            # Loop over each token in review
            for t in review:
                
                # Clean token
                cleaned_token = self.clean_token(t)

                # Check if token meets threshold
                if cleaned_token:
                    if self.threshold and self.allowed_tokens is not None:
                        if cleaned_token not in self.allowed_tokens:
                            continue
                    cleaned_review.append(cleaned_token)
            
            cleaned_corpus.append(cleaned_review)

        return np.array(cleaned_corpus, dtype='object')
    
    def fit_transform(self, tokenised_reviews:np.ndarray) -> np.ndarray:
        """fit() and transform() in one go."""
        self.fit(tokenised_reviews)
        return self.transform(tokenised_reviews)
