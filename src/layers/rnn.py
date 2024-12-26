import numpy as np
from src.initializers import Initializer, GlorotUniform, Zeros
from src.activations import Activation, Sigmoid, Tanh

class SimpleRNN:
    def __init__(
        self,
        output_size:int,
        activation:Activation,
        activation_out:Activation=None,
        weights_initializer:Initializer=GlorotUniform,
        bias_initializer:Initializer=Zeros,
        ):
        self.is_last = isinstance(activation_out, Activation)        
        self.units_out = output_size
        self.activation = activation
        self.activation_out = activation_out
        self.weights_initializer = weights_initializer()
        self.bias_initializer = bias_initializer()

    def compile(self, units_in):
        self.units_in = units_in
        
        self.w_xh = self.weights_initializer(shape=(self.units_in + self.units_out, self.units_out))
        self.b_xh = self.weights_initializer(shape=(1, self.units_out))

        if self.is_last:
            self.w_yh = self.bias_initializer(shape=(self.units_out, self.units_out))
            self.b_yh = self.bias_initializer(shape=(1, self.units_out))

        return self.units_out
    
    def forward(self, input):
        self.x = input

        self.time_steps = self.x.shape[0]
        self.batch_size = self.x.shape[1]

        self.h = np.zeros((self.time_steps, self.batch_size, self.units_out))
        self.xh = np.zeros((self.time_steps, self.batch_size, self.units_in + self.units_out))

        self.h_0 = np.zeros((self.batch_size, self.units_out))

        self.y = np.zeros((self.time_steps, self.batch_size, self.units_out)) if self.is_last else None

        for t in range(self.time_steps):
            self.xh[t] = np.concatenate((self.x[t], self.h[t-1] if t > 0 else self.h_0), axis=-1)
            self.h[t] = self.activation(self.xh[t] @ self.w_xh + self.b_xh)

            if self.is_last:
                self.y[t] = self.activation_out(self.h[t] @ self.w_yh + self.b_yh) 

        return self.y if self.is_last else self.h

    def backward(self, dout):
        self.dy = dout * (self.activation_out.backward(self.y) if self.is_last else 1)

        self.dh = np.zeros_like(self.h)
        self.dx = np.zeros_like(self.x)

        self.dw_xh = np.zeros_like(self.w_xh)
        self.db_xh = np.zeros_like(self.b_xh)

        if self.is_last:
            self.dw_yh = np.zeros_like(self.w_yh)
            self.db_yh = np.zeros_like(self.b_yh)

        for t in reversed(range(self.time_steps)):
            self.dh[t] = (self.dy[t] + ((self.dh[t+1] * self.activation.backward(self.h[t+1])) @ self.w_xh[self.units_in:].T
                    ) if t+1 < self.time_steps else self.h_0)
            
            self.dx[t] = (self.dh[t] * self.activation.backward(self.h[t])) @ self.w_xh[:self.units_in].T

            self.dw_xh += self.xh[t].T @ (self.dh[t] * self.activation.backward(self.h[t]))
            self.db_xh += np.mean(self.dh[t], axis=0)

            if self.is_last:
                self.dw_yh += (self.h[t].T @ self.dy[t]) 
                self.db_yh += np.mean(self.dy[t], axis=0)

        return self.dx


    def get_params(self):
        if self.is_last:
            params = {'w_xh': self.w_xh, 'b_xh': self.b_xh, 'w_yh': self.w_yh, 'b_yh': self.b_yh}
        else:
            params = {'w_xh': self.w_xh, 'b_xh': self.b_xh}
        return params

    def get_grads(self):
        if self.is_last:
            grads = {'w_xh': self.dw_xh, 'b_xh': self.db_xh, 'w_yh': self.dw_yh, 'b_yh': self.db_yh}
        else:
            grads = {'w_xh': self.dw_xh, 'b_xh': self.db_xh}
        return grads


class LSTM:
    def __init__(
        self,
        output_size:int,
        activations:list[Activation]=[Sigmoid, Sigmoid, Tanh, Tanh, Sigmoid],
        activation_names:list[str]=['input', 'forget', 'candidate', 'cell', 'output']
        ):
        self.units_out = output_size
        self.activations = {name:func for name, func in zip(activation_names, activations)}

    def compile(self, units_in):
        self.units_in = units_in

        self.input_gate = Gate(activation=self.activations['input'])
        self.forget_gate = Gate(activation=self.activations['forget'])
        self.candidate_gate = Gate(activation=self.activations['candidate'])
        self.output_gate = Gate(activation=self.activations['output'])

        self.input_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))
        self.forget_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))
        self.candidate_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))
        self.output_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))

        return self.units_out

    def forward(self, X):
        self.time_steps = X.shape[0]
        self.batch_size = X.shape[1]

        self.x = X
        self.h = np.zeros((self.time_steps, self.batch_size, self.units_out))

        self.fg = np.zeros((self.time_steps, self.batch_size, self.units_out))
        self.ig = np.zeros((self.time_steps, self.batch_size, self.units_out))
        self.cg = np.zeros((self.time_steps, self.batch_size, self.units_out))
        self.xh = np.zeros((self.time_steps, self.batch_size, self.units_in + self.units_out))
        self.og = np.zeros((self.time_steps, self.batch_size, self.units_out))

        self.cell = np.zeros((self.time_steps, self.batch_size, self.units_out))
        self.cell_act = np.zeros_like(self.cell)

        self.zero_memory = np.zeros((self.batch_size, self.units_out))

        for t in range(self.time_steps):
            self.xh[t] = np.concatenate((self.x[t], self.h[t-1] if t > 0 else self.zero_memory), axis=-1)     

            self.fg[t] = self.forget_gate.forward(self.xh[t]) 
            self.ig[t] = self.input_gate.forward(self.xh[t])      
            self.cg[t] = self.candidate_gate.forward(self.xh[t])
            self.og[t] = self.output_gate.forward(self.xh[t])

            self.cell[t] = self.ig[t] * self.cg[t] + self.fg[t] * (self.cell[t-1] if t > 0 else self.zero_memory)
            self.cell_act[t] = self.activations['cell']()(self.cell[t])

            self.h[t] = self.og[t] * self.cell_act[t]

        return self.h
    
    def backward(self, dout:np.ndarray):
        
        self.dy = dout

        self.dxh = np.zeros_like(self.xh)
        self.dx = np.empty_like(self.x)
        self.dh = np.zeros_like(self.h)
        self.dcell = np.zeros_like(self.cell)
        
        self.dog = np.zeros_like(self.og)
        self.dfg = np.zeros_like(self.fg)
        self.dig = np.zeros_like(self.ig)
        self.dcg = np.zeros_like(self.cg)

        self.dw_o = np.zeros_like(self.output_gate.w)
        self.dw_f = np.zeros_like(self.forget_gate.w)
        self.dw_i = np.zeros_like(self.input_gate.w)
        self.dw_c = np.zeros_like(self.candidate_gate.w)

        self.db_o = np.zeros_like(self.output_gate.b)
        self.db_f = np.zeros_like(self.forget_gate.b)
        self.db_i = np.zeros_like(self.input_gate.b)
        self.db_c = np.zeros_like(self.candidate_gate.b)

        for t in reversed(range(self.time_steps)):
            self.dh[t] = self.dy[t]
            self.dcell[t] = self.dh[t] * self.og[t] * self.activations['cell']().backward(self.cell_act[t])
            if t+1 < self.time_steps:
                self.dh[t] += self.dh[t+1] * self.dxh[t+1, :, self.units_in:]
                self.dcell[t] += self.fg[t+1] * self.dcell[t+1]

            self.dog[t] = self.dh[t] * self.cell_act[t] * self.output_gate.backward(self.og[t])
            self.dfg[t] = self.dcell[t] * (self.cell[t-1] if t > 0 else self.zero_memory) * self.forget_gate.backward(self.fg[t])
            self.dig[t] = self.dcell[t] * self.cg[t] * self.input_gate.backward(self.ig[t])
            self.dcg[t] = self.dcell[t] * self.ig[t] * self.candidate_gate.backward(self.cg[t])

            self.dxh[t] = (
                self.dog[t] @ self.output_gate.w.T + 
                self.dfg[t] @ self.forget_gate.w.T + 
                self.dig[t] @ self.input_gate.w.T +
                self.dcg[t] @ self.candidate_gate.w.T
                )

            self.dx[t] = self.dxh[t, :, :self.units_in]

            self.dw_o += self.xh[t].T @ self.dog[t]
            self.dw_f += self.xh[t].T @ self.dfg[t]
            self.dw_i += self.xh[t].T @ self.dig[t]
            self.dw_c += self.xh[t].T @ self.dcg[t]

            self.db_o += np.sum(self.dog[t], axis=0)
            self.db_f += np.sum(self.dfg[t], axis=0)
            self.db_i += np.sum(self.dig[t], axis=0)
            self.db_c += np.sum(self.dcg[t], axis=0)
        
        return self.dx
    
    def get_params(self):
        params = {
            'w_o': self.output_gate.w,
            'w_f': self.forget_gate.w,
            'w_i': self.input_gate.w,
            'w_c': self.candidate_gate.w,
            'b_o': self.output_gate.b,
            'b_f': self.forget_gate.b,
            'b_i': self.input_gate.b,
            'b_c': self.candidate_gate.b,            
            }
        return params

    def get_grads(self):
        params = {
            'w_o': self.dw_o,
            'w_f': self.dw_f,
            'w_i': self.dw_i,
            'w_c': self.dw_c,
            'b_o': self.db_o,
            'b_f': self.db_f,
            'b_i': self.db_i,
            'b_c': self.db_c,            
            }
        return params


class GRU:
    def __init__(
        self,
        output_size:int,
        activations:list[Activation]=[Sigmoid, Sigmoid, Tanh],
        activation_names:list[str]=['update', 'reset', 'candidate']
        ):
        self.units_out = output_size
        self.activations = {name:func for name, func in zip(activation_names, activations)}

    def compile(self, units_in):
        self.units_in = units_in
 
        self.update_gate = Gate(activation=self.activations['update'])
        self.reset_gate = Gate(activation=self.activations['reset'])
        self.candidate_gate = Gate(activation=self.activations['candidate'])

        self.update_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))
        self.reset_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))
        self.candidate_gate.compile(w_shape=(self.units_in + self.units_out, self.units_out), b_shape=(1, self.units_out))

        return self.units_out
    
    def forward(self, X:np.ndarray):
        self.time_steps = X.shape[0]
        self.batch_size = X.shape[1]

        self.x = X
        self.h = np.zeros((self.time_steps, self.batch_size, self.units_out))    
        self.xh = np.zeros((self.time_steps, self.batch_size, self.units_in + self.units_out))

        self.ug = np.zeros((self.time_steps, self.batch_size, self.units_out)) 
        self.rg = np.zeros((self.time_steps, self.batch_size, self.units_out)) 
        self.cg = np.zeros((self.time_steps, self.batch_size, self.units_out)) 

        self.zero_memory = np.zeros((self.batch_size, self.units_out))   

        for t in range(self.time_steps):
            self.xh[t] = np.concatenate((self.x[t], self.h[t-1] if t > 0 else self.zero_memory), axis=-1)         

            self.ug[t] = self.update_gate.forward(self.xh[t])  
            self.rg[t] = self.reset_gate.forward(self.xh[t])

            xrh = self.xh[t].copy()
            xrh[:, self.units_in:] *= self.rg[t]

            self.cg[t] = self.candidate_gate.forward(xrh)

            self.h[t] = (1 - self.ug[t]) * self.xh[t, :, self.units_in:] + self.ug[t] * self.cg[t]

        return self.h
    
    def backward(self, dout:np.ndarray):

        self.dh = dout

        self.dx = np.empty_like(self.x)
        self.dhh = np.empty_like(self.dh)

        self.dug = np.empty_like(self.ug)
        self.drg = np.empty_like(self.rg)
        self.dcg = np.empty_like(self.cg)

        self.dw_u = np.zeros_like(self.update_gate.w)
        self.dw_r = np.zeros_like(self.reset_gate.w)
        self.dw_c = np.zeros_like(self.candidate_gate.w)

        self.db_u = np.zeros_like(self.update_gate.b)
        self.db_r = np.zeros_like(self.reset_gate.b)
        self.db_c = np.zeros_like(self.candidate_gate.b)

        for t in reversed(range(self.time_steps)):
            if t+1 < self.time_steps:
                self.dh[t] += self.dhh[t+1]

            self.dcg[t] = self.dh[t] * self.ug[t] * self.candidate_gate.backward(self.cg[t])
            self.dug[t] = self.dh[t] * (self.cg[t] - self.xh[t, :, self.units_in:]) * self.update_gate.backward(self.ug[t])
            self.drg[t] = (self.dcg[t] @ self.candidate_gate.w[self.units_in:].T) * self.xh[t, :, self.units_in:] * self.reset_gate.backward(self.rg[t])

            self.dhh[t] = (
                self.dh[t] * (1 - self.ug[t]) +
                self.dug[t] @ self.update_gate.w[self.units_in:].T +
                self.drg[t] @ self.reset_gate.w[self.units_in:].T +
                self.dcg[t] @ self.candidate_gate.w[self.units_in:].T * self.rg[t]
            )

            self.dx[t] = (
                self.dcg[t] @ self.candidate_gate.w[:self.units_in].T +
                self.dug[t] @ self.update_gate.w[:self.units_in].T +
                self.drg[t] @ self.reset_gate.w[:self.units_in].T
                )
            
            self.dw_c += self.xh[t].T @ self.dcg[t] 
            self.dw_u += self.xh[t].T @ self.dug[t]
            self.dw_r += self.xh[t].T @ self.drg[t]

            self.db_c += np.sum(self.dcg[t], axis=0)
            self.db_u += np.sum(self.dug[t], axis=0)
            self.db_r += np.sum(self.drg[t], axis=0)

        return self.dx
    
    def get_params(self):
        params = {
            'w_c': self.candidate_gate.w,
            'w_u': self.update_gate.w,
            'w_r': self.reset_gate.w,
            'b_c': self.candidate_gate.b,
            'b_u': self.update_gate.b,
            'b_r': self.reset_gate.b,           
            }
        return params

    def get_grads(self):
        params = {
            'w_c': self.dw_c,
            'w_u': self.dw_u,
            'w_r': self.dw_r,
            'b_c': self.db_c,
            'b_u': self.db_u,
            'b_r': self.db_r,           
            }
        return params
    
    
class Gate:
    def __init__(
        self,
        activation:Activation,
        weights_initializer:Initializer=GlorotUniform,
        bias_initializer:Initializer=Zeros,
        ):
        self.activation = activation()
        self.weights_initializer = weights_initializer()
        self.bias_initializer = bias_initializer()

    def compile(self, w_shape:tuple, b_shape:tuple):        
        self.w = self.weights_initializer(shape=w_shape)
        self.b = self.bias_initializer(shape=b_shape)

    def forward(self, input:np.ndarray):
        z = input @ self.w + self.b
        return self.activation(z)

    def backward(self, out:np.ndarray):
        return self.activation.backward(out)