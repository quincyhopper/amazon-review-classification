import numpy as np
import pandas as pd 
from collections import defaultdict

def make_confusion_matrix(y_true:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
    """Make confusion matrix. Rows are ground truths, columns are predictions.
    
    Args:
        y_true: Array of ground truth labels.
        y_pred: Array of predicted labels.

    Returns:
        Confusion matrix (NumPy array).
    """
    # Get number of unique labels (dimensions of the confusion matrix)
    n_labels = np.unique(y_true).shape[0]

    # Initialise empty confusion matrix of shape (n_labels, n_labels)
    conf_matrix = np.zeros((n_labels, n_labels))

    # Looking at each element of y_true and y_pred together, add a 1 at the index (y_true, y_pred) in the confusion matrix
    np.add.at(conf_matrix, (y_true, y_pred), 1)

    return conf_matrix

def classification_report(conf_matrix:np.ndarray, labels:np.ndarray):
    """Generate classification report containing precision, recall, f1-score, suppport, and macro and weighted metrics. 
    
    Args:
        conf_matrix: Confusion matrix.
        labels: NumPy array of unique labels.
        round: Number of decimal places to round to.

    Returns:
        Pandas DataFrame of classification report. 
    """

    # Handle divison by zero
    epsilon = 1e-10

    # TPs are the diagonals
    TP = np.diag(conf_matrix)

    # FPs are the sums of the rows minus the TPs
    FP = np.sum(conf_matrix, axis=0) - TP

    # FNs are the sums of the columns minus the TPs
    FN = np.sum(conf_matrix, axis=1) - TP

    # Calculate metrics
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)
    support = np.sum(conf_matrix, axis=1)
    samples = np.sum(conf_matrix)

    # Accuracy is the trace divided by total samples
    accuracy = np.trace(conf_matrix) / samples

    # Calculate macro metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate weighed metrics
    weighted_precision = np.sum(precision * support) / samples 
    weighted_recall = np.sum(recall * support) / samples
    weighted_f1 = np.sum(f1 * support) / samples

    report = defaultdict()

    # Loop over each class
    for c in np.arange(conf_matrix.shape[0]):

        c = int(c)

        class_metrics = {
            'precision': f"{precision[c]}",
            'recall': f"{recall[c]}",
            'f1-score': f"{f1[c]}",
            'support': f"{support[c]}"
        }

        report[labels[c]] = class_metrics

    report['accuracy'] = {'precision': '',
                          'recall': '',
                          'f1-score': f"{accuracy}",
                          'support': samples
                          }
    
    # Add the macro average metrics to the general report
    report['macro avg'] = {'precision': f"{macro_precision}",
                           'recall': f"{macro_recall}",
                           'f1-score': f"{macro_f1}",
                           'support': samples
                           }

    # Add the weighted metrics to the report
    report['weighted avg'] = {'precision': f"{weighted_precision}",
                              'recall': f"{weighted_recall}",
                              'f1-score': f"{weighted_f1}",
                              'support': samples
                              }
    
    return pd.DataFrame(report).T