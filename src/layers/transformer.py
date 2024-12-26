import numpy as np
from src.initializers import Initializer, GlorotUniform
from src.activations import RowWiseSoftmax, Relu, Linear

class Transformer:
    def __init__(self, heads, emb_size, initializer:Initializer=GlorotUniform):
        self.heads=heads
        self.initializer=initializer
        self.emb_size=emb_size

    def compile(self, unit_in=None):
        self.attention = MultiHeadAttention(heads=self.heads, initializer=self.initializer)
        self.attention.compile(units_in=self.emb_size)
        
        self.norm1 = LayerNorm(emb_size=self.emb_size)
        self.norm2 = LayerNorm(emb_size=self.emb_size)
        self.dense1 = SubDense(input_dim=self.emb_size, out_dim=64, initializer=self.initializer, activation=Relu)
        self.dense2= SubDense(input_dim=64, out_dim=self.emb_size, initializer=self.initializer, activation=Linear)
        self.pooling = AveragePooling()

        return self.emb_size
    
    def forward(self, X:np.ndarray):
        A = self.attention.forward(X)
        out_norm1 = self.norm1.forward(X, A)
        out_hidden1 = self.dense1.forward(out_norm1)
        out_hidden2 = self.dense2.forward(out_hidden1)
        out_norm2 = self.norm2.forward(out_norm1, out_hidden2)
        y = self.pooling.forward(out_norm2)
        return y

    def backward(self, dout:np.ndarray):
        dout_pool = self.pooling.backward(dout)
        dout_norm2, dout_hidden_2 = self.norm2.backward(dout_pool)
        dout_hidden1 = self.dense2.backward(dout_hidden_2)
        dout_hidden0 = self.dense1.backward(dout_hidden1)
        dout_norm1, dA_norm = self.norm1.backward(dout_hidden0+dout_norm2)
        dA = self.attention.backward(dA_norm)
        return dA + dout_norm1
    

    def get_params(self):
        params = {
            'w_q': self.attention.w_q,
            'w_v': self.attention.w_v,
            'w_k': self.attention.w_k,
            'w_a': self.attention.w_a,
            'w_dense1': self.dense1.W,
            'b_dense1': self.dense1.b,
            'w_dense2': self.dense2.W,
            'b_dense2': self.dense2.b, 
            'gamma1': self.norm1.gamma,
            'gamma2': self.norm2.gamma,
            'beta1': self.norm1.beta,
            'beta2': self.norm2.beta
            }
        return params

    def get_grads(self):
        params = {
            'w_q': self.attention.dw_q,
            'w_v': self.attention.dw_v,
            'w_k': self.attention.dw_k,
            'w_a': self.attention.dw_a,
            'w_dense1': self.dense1.dW,
            'b_dense1': self.dense1.db,
            'w_dense2': self.dense2.dW,
            'b_dense2': self.dense2.db, 
            'gamma1': self.norm1.dgamma,
            'gamma2': self.norm2.dgamma,
            'beta1': self.norm1.dbeta,
            'beta2': self.norm2.dbeta
            }
        return params
    

class MultiHeadAttention:
    def __init__(
        self,
        heads:int,
        initializer:Initializer=GlorotUniform,
        ):
        self.heads = heads
        self.initializer = initializer()

    def compile(self, units_in:int):
        self.emb_size = units_in
        self.head_dim = self.emb_size // self.heads
        self.activation = RowWiseSoftmax()

        self.w_q = self.initializer(shape=(self.heads, self.head_dim, self.head_dim))
        self.w_k = self.initializer(shape=(self.heads, self.head_dim, self.head_dim))
        self.w_v = self.initializer(shape=(self.heads, self.head_dim, self.head_dim))
        self.w_a = self.initializer(shape=(self.emb_size, self.emb_size))

        return self.emb_size

    def forward(self, X:np.ndarray):
        '''
        Parameters
        ----------
        X : np.ndarray
            (batch_size, seq_len, emb_size)
        '''
        self.batch_size, self.seq_len, _ = X.shape
        self.X_head = X.reshape(self.batch_size, self.seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        # X (b,h,s,d), w (h,d,d)
        self.Q_head = np.matmul(self.X_head, self.w_q)
        self.K_head = np.matmul(self.X_head, self.w_k)
        self.V_head = np.matmul(self.X_head, self.w_v)
        # Q (b,h,s,d) K.T (b,h,d,s)
        H_head = np.matmul(self.Q_head, self.K_head.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        self.H_act = self.activation.forward(H_head)
        # H (b,h,s,s) V (b,h,s,d)
        self.A_head = np.matmul(self.H_act, self.V_head)  
        # (b,h,s,d) -> (b,s,h,d) -> (b,s,h*d)
        self.A = self.A_head.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.emb_size)
        # A (b,s,e) w (e,e)
        self.y = np.matmul(self.A, self.w_a) 

        return self.y
    
    def backward(self, dout:np.ndarray):
        '''
        Parameters
        ----------
        dout : np.ndarray
            (batch_size, seq_len, emb_size)
        '''
        # (b,s,e) (e,e)
        self.dA = np.matmul(dout, self.w_a.T)
        # (b,h,s,d) = (b,s,e) -> (b,s,h,d) -> (b,h,s,d)
        self.dA_head = self.dA.reshape(self.batch_size, self.seq_len, self.heads, self.head_dim).transpose(0,2,1,3)
        # A (b,h,s,d) = H (b,h,s,s) V (b,h,s,d) => dV (b,h,s,d) = dH.T (b,h,s,s), dA (b,h,s,d)
        self.dV_head = np.matmul(self.H_act.transpose(0,1,3,2), self.dA_head)
        # dH (b,h,s,s) = dA (b,h,s,d) V.T (d,h,d,s)
        dH_no_act = np.matmul(self.dA_head, self.V_head.transpose(0,1,3,2))
        self.dH_head = self.activation.backward(dH_no_act)
        # self.dH_head = np.matmul(self.dA_head, self.V_head.transpose(0,1,3,2)) * self.activation.backward(self.H_act)
        # H = Q*K.T => dQ = dH*K | (b,h,s,s)*(b,h,s,d)
        self.dQ_head = np.matmul(self.dH_head, self.K_head) / np.sqrt(self.head_dim)
        # dK = Q.T*H => (b,h,s,d) => ((b,h,d,s)*(b,h,s,s)).T
        self.dK_head = np.matmul(self.Q_head.transpose(0,1,3,2), self.dH_head).transpose(0,1,3,2) / np.sqrt(self.head_dim) 
        # y = A @ w => dw = A.T @ y (b,e,s) @ (b,s,e)
        self.dw_a = np.sum(np.matmul(self.A.transpose(0,2,1), dout), axis=0)
        # Q = X @ w => dw (h,d,d) = X.T @ Q | (b,h,d,s) @ (b,h,s,d)
        self.dw_q = np.sum(np.matmul(self.X_head.transpose(0,1,3,2), self.dQ_head), axis=0)
        self.dw_k = np.sum(np.matmul(self.X_head.transpose(0,1,3,2), self.dK_head), axis=0)
        self.dw_v = np.sum(np.matmul(self.X_head.transpose(0,1,3,2), self.dV_head), axis=0)

        self.dX_head = self.dQ_head + self.dK_head + self.dV_head
        # (b,h,s,d) -> (b,s,h,d) -> (b,s,e)
        self.dX = self.dX_head.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len, self.emb_size)

        return self.dX
    

class LayerNorm:
    def __init__(self, emb_size: int, eps: float = 1e-5):
        self.emb_size = emb_size
        self.eps = eps

        self.gamma = np.ones((emb_size,), dtype=np.float32)
        self.beta = np.zeros((emb_size,), dtype=np.float32)
        
        self.dgamma = None
        self.dbeta = None

    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self.X = X
        self.A = A
        
        self.res = X + A  

        self.mean = np.mean(self.res, axis=-1, keepdims=True)  
        self.var = np.var(self.res, axis=-1, keepdims=True)    
        self.std = np.sqrt(self.var + self.eps)                

        self.res_centered = self.res - self.mean               
        self.x_hat = self.res_centered / self.std              

        self.out = self.x_hat * self.gamma + self.beta          
        return self.out

    def backward(self, dOut: np.ndarray):
        B, S, E = dOut.shape
        N = E  

        self.dgamma = np.sum(dOut * self.x_hat, axis=(0, 1))  
        self.dbeta = np.sum(dOut, axis=(0, 1))               

        dx_hat = dOut * self.gamma                           

        dvar = np.sum(
            dx_hat * self.res_centered * (-0.5) * (self.var + self.eps) ** (-1.5),
            axis=-1, keepdims=True
        )  

        dmean = (
            np.sum(dx_hat * (-1.0 / self.std), axis=-1, keepdims=True)
            + dvar * np.mean(-2.0 * self.res_centered, axis=-1, keepdims=True)
        )  

        dres = (
            dx_hat / self.std
            + (dvar * 2.0 * self.res_centered / N)
            + (dmean / N)
        )  

        dX = dres
        dA = dres
        return dX, dA
    

class SubDense:
    def __init__(self, input_dim: int, out_dim:int,  activation=Linear, initializer: Initializer = GlorotUniform):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.initializer = initializer()
        self.W = self.initializer(shape=(self.input_dim, self.out_dim))
        self.b = np.zeros((self.out_dim,))
        self.X = None
        self.dW = None
        self.db = None
        self.activation = activation()

    def forward(self, X: np.ndarray):
        self.X = X 
        self.pre_act = np.matmul(X, self.W) + self.b
        return self.activation(self.pre_act)

    def backward(self, dout: np.ndarray):
        dout = dout * self.activation.backward(self.pre_act)
        self.dW = np.sum(np.matmul(self.X.transpose(0,2,1), dout),axis=0)
        self.db = np.sum(dout, axis=(0,1))
        dX = np.matmul(dout, self.W.T) 
        return dX
    

class AveragePooling:
    def forward(self, X: np.ndarray):
        self.X = X
        self.out = np.mean(X, axis=1) 
        return self.out

    def backward(self, dOut: np.ndarray):
        dX = dOut[:, np.newaxis, :] / self.X.shape[1]
        dX = np.repeat(dX, self.X.shape[1], axis=1)
        return dX