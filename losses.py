import numpy as np
import utils

class CrossEntropyLoss():
    """Class for Cross Entropy Loss function. 
    """
    def __init__(self):
        pass

    def compute_loss(self, logits:np.ndarray, y_true:np.ndarray) -> float:
        """Calculate cross entropy loss directly from logits.

        Uses the Log-Sum-Exp identity to bypass the softmax function, as well as maintaining numerical precision.

        Args:
            logits: Array of shape (n_samples, n_classes) containing the ouptut logits of the model.
            y_true: Array of shape (n_samples, ) of the ground truth labels (integers).

        Returns:
            Mean cross entropy loss. 
        """

        # Compute lse 
        lse = utils.logsumexp(logits).flatten() # Flatten = (n_samples, 1) -> (n_samples, )

        # Get logit of correct class for each row (n_samples, )
        correct_logits = logits[np.arange(logits.shape[0]), y_true]

        # Calculate loss (n_samples, )
        loss = lse - correct_logits

        return np.mean(loss)
    
    def backward(self, logits:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        """Calculate the gradient of the cross entropy loss function wrt the model's output.
        
        Args:
            logits: Array of shape (n_samples, n_classes) containing the ouptut logits of the model.
            y_true: Array of shape (n_samples, ) of the ground truth labels (integers).

        Returns:
            Gradient of cross entropy loss function wrt the model's output.
        """

        # Softmax
        probs = utils.softmax(logits)

        # Make one hot matrix for y_true. Shape (n_samples, classes)
        y_true_one_hot = np.zeros_like(probs)
        y_true_one_hot[np.arange(y_true.shape[0]), y_true] = 1

        # Calculate gradient
        grad = probs - y_true_one_hot # Shape (n_samples, classes)

        return grad