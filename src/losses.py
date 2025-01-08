import numpy as np

class MSE:
    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call forward before backward")
        
        batch_size = self.y_true.shape[0]
        grad = 2 * (self.y_pred - self.y_true) / batch_size
        return grad
    
class MAE:
    def __call__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean(np.abs(y_pred - y_true))
        return loss

    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call forward before backward")
        
        batch_size = self.y_true.shape[0]
        grad = np.where(self.y_pred > self.y_true, 1, -1) / batch_size
        return grad
    

class BinaryCrossEntropy:
    def __call__(self, y_true, y_pred):
        self.y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) 
        self.y_true = y_true
        return np.mean(
            -y_true * np.log(self.y_pred) - (1 - y_true) * np.log(1 - self.y_pred)
        )
    
    def backward(self):
        return -self.y_true / self.y_pred + (1 - self.y_true) / (1 - self.y_pred)
    

class DivergenceKL:
    def __call__(self, mu, log_var):
        self.mu = mu
        self.log_var = log_var
        self.var = np.exp(log_var)
        kl_loss = -0.5 * np.mean(np.sum(1 + self.log_var - self.mu**2 - self.var, axis=-1))
        return kl_loss 
    
    def backward(self):
        dmu = self.mu
        dlog_var = 0.5 * (self.var - 1)
        return dmu, dlog_var