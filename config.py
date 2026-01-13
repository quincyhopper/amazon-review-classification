CONFIG = {
        'random_state': 42,

        'data': {
            'stratify': True,
            'val_size': 0.1,
            'test_size': 0.1,
            },

        'cleaning': {
            'lowercase': True,
            'remove_punct': True,
            'remove_digits': True,
            'use_num': True,
            'remove_stop_words': True,
            'count_threshold': 5
            },

        'vectorisation': {
            'vectoriser': 'bow',
            'weighting': 'tfidf'
        },

        'model': {
            'lr': 0.01,
            'optimiser': 'adam',
            'weight_decay': 0.0,
            'n_epochs': 10,
            'batch_size': 128,
            'shuffle_training_batches': True,
            'hidden_layers': [],
            },
    }