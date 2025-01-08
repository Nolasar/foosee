import numpy as np
from src.initializers import Initializer, GlorotUniform, Zeros
from src.activations import Activation, Linear

class Dense:
    """
    Fully-connected layer implementation.

    Parameters
    ----------
    units : int
        Number of neurons in the layer.
    activation : str
        Activation function to use.
    weights_initializer : str, optional, default='glorot_uniform'
        Initializer for the weights matrix.
    bias_initializer : str, optional, default='glorot_uniform'
        Initializer for the bias vector.
    """
    def __init__(
        self,
        units: int,
        activation:Activation=Linear(),
        weights_initializer:Initializer=GlorotUniform,
        bias_initializer:Initializer=Zeros,
        random_state:int = 42
    ):
        """
        Initialize the Dense layer.
        """
        self.units = units
        self.activation = activation
        self.weights_initializer = weights_initializer()
        self.bias_initializer = bias_initializer()
        self.rnd_state = random_state

    def compile(self, units_in):
        """
        Initialize weights and biases based on input size.

        Parameters
        ----------
        input_size : int or tuple
            Number of input features or neurons in the previous layer.
            If a tuple is provided, it supports flattened Conv2D output.
        """
        self.units_in = units_in

        # Initialize weights and bias
        self.weights = self.weights_initializer(shape=(units_in, self.units),  rnd_state = self.rnd_state)
        self.bias = self.bias_initializer(shape=(1, self.units), rnd_state = self.rnd_state)
        return self.units

    def forward(self, input: np.ndarray):
        """
        Perform the forward pass of the Dense layer.

        Parameters
        ----------
        input : np.ndarray
            Input data of shape (samples, input_units).

        Returns
        -------
        out : np.ndarray
            Output of the layer after applying weights, bias, and activation.
            Shape: (samples, units).
        """
        self.input = input  # Save input for backward pass
        # Linear combination
        self.output = self.input @ self.weights + self.bias
        # Apply activation function if specified
        if self.activation is not None:
            self.output = self.activation(self.output)
        return self.output

    def backward(self, prev_grad: np.ndarray):
        """
        Perform the backward pass to compute gradients and propagate them.

        Parameters
        ----------
        prev_grad : np.ndarray
            Gradient of the loss with respect to the layer's output.

        Returns
        -------
        curr_grad : np.ndarray
            Gradient of the loss with respect to the layer's input.
        """
        # Compute gradient of activation
        grad = prev_grad * self.activation.backward(self.output)
        # Compute gradients for weights and biases
        if len(self.input.shape) <= 2:
            self.dweights = self.input.T @ grad
            self.dbias = np.sum(grad, axis=0, keepdims=True)
        else: 
            # (b, f_in, n) @ (b , n, f_out) = sum_b [(b, f_in, f_out)]
            self.dweights = np.sum(self.input.transpose(0,2,1) @ grad, axis=0)
            # self.dweights = np.einsum('bin,bno->io, self.input, grad)
            self.dbias = np.sum(grad, axis=(0,1), keepdims=False)
        # Compute gradient for the previous layer
        return grad @ self.weights.T

    def get_params(self):
        """
        Get the trainable parameters of the Dense layer.

        Returns
        -------
        dict
            Dictionary containing {'weights': weights, 'bias': bias}.
        """
        return {'weights': self.weights, 'bias': self.bias}

    def get_grads(self):
        """
        Get the gradients of the trainable parameters.

        Returns
        -------
        dict
            Dictionary containing {'weights': dweights, 'bias': dbias}.
        """
        return {'weights': self.dweights, 'bias': self.dbias}

