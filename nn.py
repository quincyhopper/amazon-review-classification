import numpy as np
import pickle

class FeedForwardNetwork():
    """ Class for a feed forward network consisting of an arbitrary number of layers of an arbitrary size.

    This class is capable of implementing a multi-layer perceptron or, if an empty list is passed to `hidden_layers`, a multinomial/binary logistic regression. It handles backpropagation automatically.

    Attributes:
        weights (list[np.ndarray]): List of the weight matrices for each layer. For example, weights[0] is the weight matrix in the 'input layer' and weight[-1] is the weight matrix of the last hidden layer.
        biases (list[np.ndarray]): List of the bias vectors for each layer. Same order as weights list.
        layers (list[int]): List of integers corresponding to the number of units in each layer.
        n_params (int): Total numnber of weights and biases in model.
        layer_inputs (list[np.ndarray]): List of the input to each layer during the forward pass. For example, the input to the last hidden layer is the post-activation output of the previous hidden layer. Stored for backpropagation.
        layer_zeds (list[np.ndarray]): List of the linear output (pre-activation) of every layer. Stored for backpropagation.
    """

    def __init__(self, input_dim:int, hidden_layers:list[int], output_dim:int, random_state:int=None):
        """Initialise feed forward network.
        
        Args:
            input_dim: Dimensions (no. of columns) of input matrix.
            hidden_layers: List containing the number of neurons in each hidden layer. If empty, the model is a logistic regression.
            output_dim: The number of classes to output predictions for.
        """

        self.weights = []
        self.biases = []
        self.layers = [input_dim] + hidden_layers + [output_dim]
        self.rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

        # Initialise correctly shaped weights and biases for every layer
        for i in range(len(self.layers) - 1):
            in_size = self.layers[i]
            out_size = self.layers[i+1]

            # Kaiming initialisation if neural network
            if len(self.layers) > 2:
                std = np.sqrt(2 / in_size) 
            else:
                std = 0.01

            w = self.rng.normal(0, std, size = (in_size, out_size))
            b = np.zeros((1, out_size))

            self.weights.append(w)
            self.biases.append(b)

        # Count parameters
        self.n_params = sum([w.size for w in self.weights]) + sum([b.size for b in self.biases])

    def forward(self, x:np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network.

        Args:
            x: The input data. Has shape (n_samples, n_dimensions), where n_dimensions is the number of input dimensions defined at initialisation.

        Returns:
            The logits of the output layer. Has shape (n_samples, n_classes).
        """

        self.layer_inputs = [] # Store layer inputs for backprop
        self.layer_zeds = []   # Store linear outputs for backprop
        output_layer = len(self.weights) - 1 # Index of last layer

        a = x 

        for layer, (w, b) in enumerate(zip(self.weights, self.biases)):
            self.layer_inputs.append(a) # Store input

            # Compute linear output
            z = np.dot(a, w) + b
            self.layer_zeds.append(z) # Store linear output

            # Only apply ReLU if NOT the output layer
            if layer == output_layer:
                a = z
            else:
                a = np.maximum(z, 0) # ReLU activation
        
        return a
    
    def backward(self, gradient:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform backpropagation through an arbitrary number of layers. The activation function is hard-coded to ReLU.

        To allow the user to employ any loss function, the gradent of the model's output wrt the loss function is computed externally by the loss class. This is the starting point for backpropagation, and is initially set as `current_grad`. This method then loops backwards through each layer and, in each loop, calculates the following derivaties
            (1) The gradient of the loss function wrt the layer's linear output (applying dropout masks if necessary).
            (2) The gradient of the loss function wrt the layer's weights.
            (3) The gradient of the loss function wrt the layer's biases.
            (4) The gradient of the loss function wrt the current layer's input.

        The gradient (4) is set to `current_grad` and the next layer uses this as the start of its chain.

        Args:
            gradient: The gradient of the loss function wrt the model's output (calculated by the loss class).

        Returns:
            Two lists containing the gradients of the loss function wrt the weights and biases for each layer in ascending layer order, such that the first element in both lists corresponds to the gradients of the input layer.
        """
        
        # Store gradients
        w_grads = []
        b_grads = []

        # First layer in loop will be the output layer
        output_layer = True

        # Gradient of loss function wrt to 
        current_grad = gradient

        # Loop backwards through layers
        for W, x, z in reversed(list(zip(self.weights,
                                            self.layer_inputs, 
                                            self.layer_zeds
                                            ))):
            
            # Skip ReLU derivative if output layer
            if output_layer:
                output_layer = False
            else:
                # Gradient (1)
                current_grad = current_grad * (z > 0)

            # Gradient (2)
            dw = np.dot(x.T, current_grad)
            w_grads.append(dw) # Store gradient wrt weight

            # Gradient (3)
            db = np.sum(current_grad, axis=0, keepdims=True)
            b_grads.append(db) # Store gradient wrt bias

            # Gradient (4)
            current_grad = np.dot(current_grad, W.T)

        # Put gradients back into normal order
        return w_grads[::-1], b_grads[::-1]

    def predict(self, x:np.ndarray) -> np.ndarray:
        """Predict the class of input data.
        
        Args:
            x: The input data. Has shape (n_samples, n_dimensions).

        Returns:
            A vector of integers corresponding to the predicted class for each sample. Has shape (n_samples, ).
        """

        logits = self.forward(x)

        # Predictions are the index of the largest logit for each review
        return np.argmax(logits, axis=1)
    
    def state_dict(self):
        """Return dictionary of current weights and biases. Can be used to reconstruct model."""
        return {
            'weights': list(self.weights),
            'biases': list(self.biases),
            'layers': self.layers
        }
    
    def save(self, filename, verbose=False):
        """Save model weights, biases and layers to pickle file."""

        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)

        if verbose:
            print(f'Model saved as {filename}')

    def load(self, new_state_dict:dict) -> None:
        """Load a given model state dictionary into an existing model instance.
        
        Args:
            new_state_dict (dict): State dictionary from previously trained model instance.
        """
        
        for layer, _ in enumerate(self.weights):
            new_weight = new_state_dict['weights'][layer]
            self.weights[layer][:] = new_weight # Assign new weights in-place

        for layer, _ in enumerate(self.biases):
            new_bias = new_state_dict['biases'][layer]
            self.biases[layer][:] = new_bias # Assign new bias in-place