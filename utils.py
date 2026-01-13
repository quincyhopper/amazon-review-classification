"""Functions for:
    - Loading raw data
    - Train/test/val splitting
    - Balancing training set
    - Loading batches
    - Loading a trained model
    - Saving training logs
    - Saving confusion matrix
    - Loading the vocab list from a saved BagOfWords vocab list
    - Computing the log-sum-exp of logits
    - Computing softmax
"""

import csv
import numpy as np
import pandas as pd 
import pickle
from nn import FeedForwardNetwork

def load_raw_data() -> tuple[list, list]:
    """Extract reviews and labels from saved CSV file.
    
    Returns:
        reviews and labels.
    """

    reviews = []
    labels = []

    with open('data/reviews.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header row
        for row in reader:
            reviews.append(row[0])
            labels.append(row[1])

    return reviews, labels

def train_val_test_split(labels:np.ndarray, 
                     stratify:bool=True,
                     val_size: float=0.1,
                     test_size:float=0.1, 
                     random_state:int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given array of target labels, return the indices for train/val/test splits using specified proportions.

    Args:
        labels: np.darray of the encoded target labels
        stratify: If True, the proportion of each class is maintained across all splits. For example, if class A makes up 20% of the entire dataset, each train/val/test split will be 20% class A. If `stratify=False`, data will just be randomly split into specificed sizes.
        val_size: relative size of validation split.
        test_size: relative size of test split.
        random_state: RNG seed.

    Returns: 
        train, validation and test indices.
    """
    
    rng = np.random.default_rng(random_state)

    if stratify:
        # Get sorted indices of labels
        sorted_idx = np.argsort(labels)

        # Use sorted indices to create an array of sorted labels
        sorted_labels = labels[sorted_idx]

        # Get the index where each class starts
        unique_labels, start_idx = np.unique(sorted_labels, return_index=True)

        # Split sorted indices on class
        class_idx = np.split(sorted_idx, start_idx[1:])

        # Make dictionary where keys=class and values=indices of said class
        label_indices = dict(zip(unique_labels, class_idx))

        # Initialise lists for the splits
        train_idx_chunks = []
        val_idx_chunks = []
        test_idx_chunks = []

        # Loop over the indices of each class
        for idx in label_indices.values():

            # Randomly shuffle the class' indices
            rng.shuffle(idx)

            # Compute proportions
            # This is where the relative proportion of each class is extracted 
            total = len(idx) # Total count of class 
            val_count = int(total * val_size)
            test_count = int(total * test_size)
            train_count = total - val_count - test_count # Train size is the remainder

            # Append proportional quantities of each class to the training indices
            train_idx_chunks.append(idx[:train_count])
            val_idx_chunks.append(idx[train_count:train_count+val_count])
            test_idx_chunks.append(idx[train_count+val_count:])

        # Join the split together
        train_idx = np.concatenate(train_idx_chunks)
        val_idx = np.concatenate(val_idx_chunks)
        test_idx = np.concatenate(test_idx_chunks)

        # Shuffle again (since it's currently grouped by class)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    else:
        # Compute proportions
        total = len(labels)
        val_count = int(total * val_size)
        test_count = int(total * test_size)
        train_count = total - val_count - test_count

        # Shuffle the indices of the data
        idx = np.arange(len(labels))
        rng.shuffle(idx)

        # Split on split sizes
        train_idx = idx[:train_count]
        val_idx = idx[train_count:train_count+val_count]
        test_idx = idx[train_count+val_count:]
        
    return train_idx, val_idx, test_idx

def loader(X:np.ndarray, y:np.ndarray, batch_size:int=32, shuffle:bool=False, random_state:int=None):
    """Separates input and label data into batches.

    Args:
        X: Review data.
        y: Label data.
        batch_size: Size of each batch.
        shuffle: If `True`, indices are shuffled before batches are calculated. 
        random_state: Random state for shuffling.

    Yields:
        Batches of input and label data.
    """

    # Compute total number of samples
    n_samples = X.shape[0]

    # Get vector of row numbers
    idx = np.arange(n_samples)

    # Initialise random state for shuffling batches
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    
    # Loop over each batch
    for start_idx in range(0, len(idx), batch_size):

        # Calculate end index
        end_idx = min(start_idx+batch_size, n_samples)

        # Extract indices of batch
        batch_idx = idx[start_idx:end_idx]

        yield X[batch_idx], y[batch_idx]

def load_model(filename:str) -> FeedForwardNetwork:
    """Load model parameters."""

    # Read file
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)

    # Extract layers
    layers = state_dict['layers']

    # Extract input dimensions, output dimensions, and any hidden layer dimensions
    input_dim = layers[0]
    output_dim = layers[-1]
    if len(layers) > 2:
        hidden_layers = layers[1:-1]
    else:
        hidden_layers = []

    # Initialise model
    model = FeedForwardNetwork(input_dim, hidden_layers, output_dim)

    # Fill in weights and biases from saved model 
    model.weights = state_dict['weights']
    model.biases = state_dict['biases']

    return model

def save_training_log(training_log:dict, filename:str) -> None:
    """Save training log to .csv file.
    
    Args:
        training_log: Dictionary of training log. Keys are epochs, values are train loss, val loss and val accuracy.
    """

    df = pd.DataFrame(training_log).T
    df.index.name = "Epoch"
    df.to_csv(filename)
    print(f"Training log saved as {filename}")

def save_conf_matrix(conf_matrix:np.ndarray, filename:str, labels:np.ndarray|list) -> None:
    """Save confusion matrix to .csv file.
    
    Args:
        conf_matrix: NumPy array of confusion matrix.
        filename: Name of file to save to.
        Labels: NumPy array or list of unique labels.
    """

    df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    df.to_csv(filename, index=True)

def load_vocab_dict(filename:str) -> dict:
    """Load vocab dictionary of BagOfWords vectoriser.
    
    Args:
        filename: .pkl file of vocab dictionary.
    """

    with open (filename, 'rb') as f:
        vocab_dict = pickle.load(f)
    return vocab_dict

def logsumexp(logits:np.ndarray) -> np.ndarray:
    """Compute the logarithm of the denominator of the softmax function.
    
    By using the maximum value in a row as a constant, the largest exponent will never be more than e^0, preventing over/underflow.

    Args:
        logits (np.ndarray): NumPy array with shape (n_samples, n_classes) where each element is the logit for a given review and class.

    Returns:
        NumPy matrix of shape (n_samples, ) where each element is the logarithm of the sum of the exponentiated elements in the row.
    """
    c = np.max(logits, axis=1, keepdims=True)
    return c + np.log(np.sum(np.exp(logits - c), axis=1, keepdims=True))

def softmax(logits:np.ndarray) -> np.ndarray:
    """Compute softmax probabilities using logsumexp.
    
    Args:
        logits: NumPy array with shape (n_samples, n_classes) where each element is the logit for a given review and class.

    Returns:
        NumPy array with shape (n_samples, n_classes) where each element is the probability of a class for a given review. 
    """
    lse = logsumexp(logits)
    return np.exp(logits - lse)
