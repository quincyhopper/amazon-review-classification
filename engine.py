import numpy as np

def train(model, loader, criterion, opt):
    """Perform a training loop over all batches (forward pass, backwards pass). Constitutes one whole epoch.

    Args:
        model (FeedForwardNetwork): Instance of a model class.
        loader: Generator containing the batches of x and y.
        criterion: Instance of a loss function class, such as CrossEntropyLoss.
        opt: Instance of an opt class, such as Adam.

    Returns:
        avg_loss (float): The mean loss over all batches in epoch.
        avg_acc (float): The mean accuracy over all batches in epoch.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0

    # Loop over batches
    for x, y in loader:

        # Increment batch counter
        num_batches += 1

        # Forward pass
        logits = model.forward(x)

        # Calculate batch loss
        batch_loss = criterion.compute_loss(logits, y)
        epoch_loss += batch_loss

        # Calculate batch accuracy
        batch_preds = np.argmax(logits, axis=1)
        batch_acc = (batch_preds == y).mean()
        epoch_acc += batch_acc

        # Calculate gradient wrt output
        grad_output = criterion.backward(logits, y_true=y)

        # Calculate gradients for parameters
        w_grads, b_grads = model.backward(grad_output)

        # Update parameters
        opt.step(w_grads, b_grads)

    # Calculate the mean batch loss 
    avg_loss = epoch_loss / num_batches

    # Calculate mean batch accuracy
    avg_acc = epoch_acc / num_batches

    return avg_loss, avg_acc


def eval(X:np.ndarray, y_true:np.ndarray, model, criterion):
    """Evaluate the model on a validation/test set.

    Args:
        X: Input data (one hot encoded matrix of reviews).
        y_true: Target labels.
        model: Instance of a model class, such as MLP.
        criterion: Instance of a loss function class, such as CrossEntropyLoss.

    Returns:
        loss: Model's loss on the validation/test set.
        acc: Model's accuracy on the validation/test set.
        y_pred: Model's predictions. Used for building confusion matrix.
    """

    # Forward pass
    logits = model.forward(X)

    # Compute loss
    loss = criterion.compute_loss(logits, y_true)

    # Make prediction
    y_pred = model.predict(X)

    # Get accuracy
    acc = (y_pred == y_true).mean()

    return loss, acc, y_pred
