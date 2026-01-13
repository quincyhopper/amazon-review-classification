import numpy as np
from collections import defaultdict

import utils
import preprocessing
import encoder
import nn
import losses
import optim
import engine
import metrics
import config

def get_datasets(config:dict):
    """Load, split, tokenise, clean and vectorise data.
    
    Args:
        config: Dictionary that at least contains a nested dictionary called 'data'. Contains splitting and cleaning parameters. 

    Returns:
        Dictionary containing all datasplits and the unique labels.
    """

    # Load data
    reviews, labels = utils.load_raw_data()

    # Tokenise reviews
    tokenised_reviews = preprocessing.tokenise(reviews)

    # Encode labels via inverse, compute counts of each class
    unique_labels, y = np.unique(labels, return_inverse=True)

    # Get split indices (unpack 'data' dictionary settings)
    train_idx, val_idx, test_idx = utils.train_val_test_split(y, **config['data'], random_state=config['random_state'])

    # Split labels into train/val/test
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Split reviews into train/val/test
    train_reviews, val_reviews, test_reviews = tokenised_reviews[train_idx], tokenised_reviews[val_idx], tokenised_reviews[test_idx]

    # Clean data
    cleaner = preprocessing.Cleaner(**config['cleaning'])
    train_clean = cleaner.fit_transform(train_reviews)
    val_clean = cleaner.transform(val_reviews)
    test_clean = cleaner.transform(test_reviews)

    # Get TF-IDF weights 
    if config['vectorisation']['weighting'] is not None:
        tfidf = encoder.TFIDF()
        tfidf.learn_idf(train_clean)
        tfidf.save('idf.pkl')
    else:
        tfidf = None

    # Vectorise cleaned reviews
    if config['vectorisation']['vectoriser'] == 'bow':
        vectoriser = encoder.BagOfWords(tfidf=tfidf)
    elif config['vectorisation']['vectoriser'] == 'glove':
        vectoriser = encoder.GloveVectoriser(tfidf=tfidf)
    else:
        raise ValueError('No vectoriser was given.')
    X_train = vectoriser.fit_transform(train_clean)
    X_val = vectoriser.transform(val_clean)
    X_test = vectoriser.transform(test_clean)

    # Save vocab list if BagOfWords
    if isinstance(vectoriser, encoder.BagOfWords):
        vectoriser.save('vectoriser.pkl')

    # Package all preprocessing into one dictionary to return
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'labels': unique_labels,
    }

    return data, vectoriser, cleaner

def train_model(config, data):
    """Initialise model, loss function, optimiser and learning rate scheduler before running full training procedure.
    
    Args:
        config: Dictionary that at least contains nested dictionary called 'model'. Contains model hyperparameters.
        data: Dictionary containing data splits and unique labels.

    Returns:
        Class instance of model after training.
    """

    model_config = config['model']
    
    # Initialise model
    input_dim = data['X_train'].shape[1] # Dimensionality
    output_dim = len(data['labels'])
    model = nn.FeedForwardNetwork(input_dim=input_dim, 
                                  hidden_layers=model_config['hidden_layers'], 
                                  output_dim=output_dim, 
                                  random_state=config['random_state'])

    # Initialise loss function
    criterion = losses.CrossEntropyLoss()

    # Initialise optimiser (Adam or SGD)
    if model_config['optimiser'] == 'adam':
        opt = optim.Adam(model.weights, model.biases, lr=model_config['lr'], weight_decay=model_config['weight_decay'])
    else:
        opt = optim.SGD(model.weights, model.biases, lr=model_config['lr'], weight_decay=model_config['weight_decay'])

    # Initialise learning rate scheduler
    steplr = optim.LRScheduler(model=model, opt=opt, lr_patience=5, es_patience=10)

    # Initialise dictionary for saving training log
    training_log = defaultdict(dict)

    # TRAINING PROCEDURE
    print(f"Training model with  {model.n_params:,} parameters.")
    for epoch in range(model_config['n_epochs']):

        # Get batches for training data
        train_loader = utils.loader(data['X_train'], 
                                    data['y_train'], 
                                    batch_size=model_config['batch_size'], 
                                    shuffle=model_config['shuffle_training_batches'], 
                                    random_state=config['random_state'])

        # Train model
        train_loss, train_acc = engine.train(model=model, loader=train_loader, criterion=criterion, opt=opt)

        # Evaluate model on validation set
        val_loss, val_acc, _ = engine.eval(data['X_val'], data['y_val'], model=model, criterion=criterion)

        # Print stats
        print(f"Epoch [{epoch + 1}/{model_config['n_epochs']}] | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
        
        # Save training log
        training_log[epoch+1] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc, 
            "val_acc": val_acc,
            "lr": opt.lr
        }

        # Break loop if scheduler doesn't see improvement
        stop = steplr.step(val_loss, epoch)
        if stop:
            print("Early stopping triggered because model is not improving.")
            break

    print("Training complete.")

    # Save model
    model.load(steplr.best_params)
    model.save('model_params.pkl', verbose=True)
    print(f"Best model at epoch {steplr.best_epoch} with val loss {steplr.best_loss:.4f}.")

    # Save training log
    utils.save_training_log(training_log, 'training_log.csv')

    return model

def evaluate_model(data, model, on:str, save=False):
    """Perform prediction on either validation or test set. Save confusion matrix. Print and save classification report
    
    Args:
        data: Dictionary containing data splits and unique labels.
        model: Instance of trained model.
        on: Specifies which split to perform prediction on. Can be 'val' or 'test'.
    """

    if on not in ['val', 'test']:
        raise ValueError("'on' must be 'val' or 'test'.")

    if on == 'val':
        X = data['X_val']
        y = data['y_val']
    else:
        X = data['X_test']
        y = data['y_test']

    # Initialise loss function
    criterion = losses.CrossEntropyLoss()

    loss, acc, y_pred = engine.eval(X, y, model=model, criterion=criterion)
    
    print(f"\n{on} loss: {loss:.4f} | {on} acc: {acc:.4f}")
    
    conf_matrix = metrics.make_confusion_matrix(y, y_pred)
    report = metrics.classification_report(conf_matrix, data['labels'])
    print(f"\n{on} classification report:\n", report)

    if save:
        utils.save_conf_matrix(conf_matrix, filename='confusion_matrix.csv', labels=data['labels'])
        report.to_csv('classification_report.csv')

if __name__ == '__main__':
    
    config = config.CONFIG

    data, vectoriser, cleaner = get_datasets(config)

    model = train_model(config, data)

    evaluate_model(data, model, on='val')