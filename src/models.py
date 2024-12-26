import numpy as np
from src.optimizers import Adam
from tqdm import tqdm

class Sequential:
    def __init__(self, *args):
        self.layers = [*args]
        self.is_compiled = False
        self.history = {
            'loss': [], 
            'metric': []
            }


    def compile(self, input_size):
        self.is_compiled = True
        units_in = input_size
        for layer in self.layers:
            units_in = layer.compile(units_in)


    def _feedforward(self, X): 
        output = X 
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    
    def _backprop(self, dloss):
        gradient = dloss
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


    def fit(
        self, 
        X:np.ndarray, 
        y:np.ndarray,
        loss_function, 
        batches=None,
        optimizer=Adam,
        epochs:int=50, 
        learning_rate:float=.001, 
        ):

        if not self.is_compiled:
            self.compile(X.shape[-1])

        optimizer = optimizer(self.layers, lr=learning_rate)

        if batches is None:
            batches = zip(X,y)
        
        num_batches = len(batches)
        for epoch in range(epochs):

            epoch_loss = 0
            with tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for batch_X, batch_y in batches:
                    y_pred = self._feedforward(batch_X)
                    loss = loss_function(y_true=batch_y, y_pred=y_pred)
                    epoch_loss += loss

                    dloss = loss_function.backward()

                    self._backprop(dloss)

                    optimizer.step()

                    pbar.set_postfix({'Loss': loss})
                    pbar.update(1)

                epoch_loss /= num_batches

                self.history['loss'].append(epoch_loss)

            print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}')


    def predict(self, X:np.ndarray):
        return self._feedforward(X)
    
    
