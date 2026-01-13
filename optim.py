import numpy as np
import copy

class Adam():
    """Class for Adam optimisation, which computes adaptive learning rates for each parameter. 
    
    Calculates moving averages of gradients (1st moment) and squared gradients (2nd moment). Also calculates a bias-corrected learning rate for each time step, effectively decreasing the learning rate over time.
    
    Attributes:
        t (int): Iteration counter. 
        m_w (list[np.ndarray]): Moving average of weight gradients.
        v_w (list[np.ndarray]): Moving average of squared weight gradients.
        m_b (list[np.ndarray]): Moving average of bias gradients.
        v_b (list[np.ndarray]): Moving average of squared bias gradients.
        beta_1 (float): Decay rate for 1st moment; 0.9 by default.
        beat_2 (float): Decay rate for 2nd moment; 0.999 by default.
        epsilon (float): Small value added to denominator to prevent division by zero; 1e-08 by default. 
        
    """
    def __init__(self, weights:list[np.ndarray], biases:list[np.ndarray], lr:float, weight_decay:float):
        """Initialise Adam optimiser.
        
        Args:
            weights: List of NumPy arrays of weight matrices.
            biases: List of NumPy arrays of bias vectors.
            lr: Global learning rate.
            weight_decay: L2 penalty coefficient. If set to 0.0, no penalty is applied. 
        """

        self.weights = weights
        self.biases = biases
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = 0.9
        self.b2 = 0.999
        self.epsilon = 1e-08
        self.t = 0

        self.m_w = [np.zeros_like(w) for w in self.weights] # Mean estimate for weights
        self.v_w = [np.zeros_like(w) for w in self.weights] # Variance estimate for weights
        self.m_b = [np.zeros_like(b) for b in self.biases]  # Mean estimate for biases
        self.v_b = [np.zeros_like(b) for b in self.biases]  # Variance estimate for biases

    def step(self, w_grads:list[np.ndarray], b_grads:list[np.ndarray]) -> None:
        """Update weights and biases.
        
        Args:
            w_grads: List of NumPy arrays of weight gradients.
            b_grads: List of NumPy arrays of bias gradients.
        """

        # Increment time step
        self.t += 1

        # Compute bias-corrected learning rate for current time step
        self.lr_t = self.lr * (np.sqrt(1 - (self.b2 ** self.t)) / (1 - (self.b1 ** self.t)))

        # Loop over weights
        for w, dw, m, v in zip(self.weights, w_grads, self.m_w, self.v_w):
            
            # Update first moment estimate
            m[:] = (self.b1 * m) + ((1 - self.b1) * dw)
            
            # Update second moment estimate
            v[:] = (self.b2 * v) + ((1 - self.b2) * np.square(dw))

            # Apply L2 penalty before updating weight
            if self.weight_decay > 0:
                w -= self.lr_t * self.weight_decay * w

            # Update weights
            w -= (self.lr_t * m) / ((np.sqrt(v)) + self.epsilon)

        # Loop over biases
        for b, db, m, v in zip(self.biases, b_grads, self.m_b, self.v_b):

            # Update first moment estimate
            m[:] = (self.b1 * m) + ((1 - self.b1) * db)
            
            # Update second moment estimate
            v[:] = (self.b2 * v) + ((1 - self.b2) * np.square(db))

            # Update biases
            b -= (self.lr_t * m) / ((np.sqrt(v)) + self.epsilon)

class SGD():
    """Class for optimising model parameters with stochastic gradient descent.
    
    Attributes:
        momentum: Decay coefficient for weighted moving average of gradients; 0.9 by default. 
        m_w (list[np.ndarray]): Moving average of weight gradients.
    """
    def __init__(self, weights:list[np.ndarray], biases:list[np.ndarray], lr:float, weight_decay:float=0.0):
        """Initialise SGD optimiser.
        
        Args:
            weights: List of NumPy arrays of weight matrices.
            biases: List of NumPy arrays of bias vectors.
            lr: Global learning rate.
            weight_decay: L2 penalty coefficient. If set to 0.0, no penalty is applied. 
        """

        self.weights = weights
        self.biases = biases
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = 0.9
        self.m_w = [np.zeros_like(w) for w in self.weights]

    def step(self, w_grads:list, b_grads:list) -> None:
        """Update weights and biases.
        
        Args:
            w_grads: List of NumPy arrays of weight gradients.
            b_grads: List of NumPy arrays of bias gradients.
        """

        # Loop over weights
        for w, dw, m in zip(self.weights, w_grads, self.m_w):
            
            # Add penalty to gradient
            if self.weight_decay > 0:
                dw = dw + (self.weight_decay * w)

            # Calculate velocity
            m[:] = self.momentum * m + dw

            # Update weight
            w -= (self.lr * m)

        # Loop over biases
        for b, db in zip(self.biases, b_grads):
            b -= self.lr * db

class LRScheduler():
    """Class for scheduling learning rate and implementing early stopping.
    
    Attributes:
        model: The model whose state_dict is saved if loss improves.
        opt: The optimiser whose learning rate is adjusted.
        lr_count (int): Counter for epochs without loss improvement. Used for LR decay. 
        es_count (int): Counter for epochs without loss improvement. Used for early stopping.
        best_loss (float): Lowest validation loss observed so far.
        best_params (dict): The parameters of the model when the lowest validation loss was observed.
        best_epoch (int): Epoch at which lowest validation loss was observed. Used to print when training stopped. 
    """
    def __init__(self, model, opt, lr_patience=None, es_patience=None):
        """Initialise learning rate scheduler.
        
        Args:
            model: Model instance. 
            opt: Optimiser instance. 
            lr_patience: Number of epochs to go without improvement before decaying learning rate.
            es_patience: Number of epochs to go without improvement before stopping training.
        """
        
        self.model = model
        self.opt = opt
        
        self.lr_patience = lr_patience
        self.es_patience = es_patience
        self.lr_count = 0
        self.es_count = 0

        self.best_loss = float('inf')
        self.best_params = None
        self.best_epoch = None

    def step(self, val_loss:float, epoch:int, verbose=True) -> None:
        """Check if validation loss is improving and trigger learning rate decay or early stopping if not.
        
        Args:
            val_loss: Validation loss at current epoch.
            epoch: Current epoch.
            verbose: If True, prints 'Learning rate decreased to {new learning rate}'.
        """
        
        # Small margin by which we consider a new loss value an improvement
        delta = 1e-6
        
        if val_loss < self.best_loss - delta:
            # Save best model
            self.best_params = copy.deepcopy(self.model.state_dict())
            self.best_epoch = epoch+1 # Save new best epoch
            self.best_loss = val_loss # Save new best loss
            self.lr_count = 0         # Reset lr_count
            self.es_count = 0         # Reset es_count
        else:
            self.lr_count += 1
            self.es_count += 1
            
            # Return 'stop training' signal if early stopping is triggered
            if self.es_count == self.es_patience:
                return True

            # Decrease learning rate by an order of magnitude if patience wears thin
            if self.lr_count == self.lr_patience:
                self.opt.lr *= 0.1
                self.lr_count = 0 # Reset patience counter
                if verbose:
                    print(f"Learning rate decreased to {self.opt.lr}")