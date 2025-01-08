import numpy as np

class Activation:
    def __init__(self):
        pass

class Linear(Activation):
    def __call__(self, x):
        return x
    
    def backward(self, out):
        return np.ones_like(out)
        
class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, out):
        return out * (1 - out)

class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)

    def backward(self, out):
        return 1 - out**2

class Relu(Activation):
    def __call__(self, x):
        return np.maximum(x, 0)

    def backward(self, out):
        return (out > 0).astype(float)

class LeakyRelu:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.output = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        dx = d_out.copy()
        dx[self.output <= 0] *= self.alpha
        return dx

class RowWiseSoftmax(Activation):
    def __init__(self, axis=-1):
        self.axis = axis
        self.out_2d = None
        self.original_shape = None

    def forward(self, x):
        """
        x: (B, H, S, S) - тензор, где последняя ось = S это "колонки",
           а по ним нужно применять row-wise softmax.
        Возвращает тот же shape (B, H, S, S) с row-wise softmax построчно.
        """
        self.original_shape = x.shape    # (B, H, S, S)
        # Шаг 1: "сплющиваем" в (B*H*S, S)
        x_2d = x.reshape(-1, x.shape[-1])  # => (N, S), где N = B*H*S

        # Шаг 2: вычитаем max для численной стабильности
        x_2d_max = np.max(x_2d, axis=1, keepdims=True)  # (N, 1)
        e_x = np.exp(x_2d - x_2d_max)                    # (N, S)
        sum_e_x = np.sum(e_x, axis=1, keepdims=True)     # (N, 1)

        # Шаг 3: вычисляем 1D-softmax
        out_2d = e_x / sum_e_x                           # (N, S)

        # Запоминаем out_2d для backward
        self.out_2d = out_2d

        # Возвращаем в форму (B, H, S, S)
        return out_2d.reshape(self.original_shape)

    def backward(self, dOut):
        """
        dOut: градиент от следующего слоя, shape (B, H, S, S).
        Возвращаем dX той же формы (B, H, S, S).
        """
        # (B,H,S,S) -> (N, S)
        dOut_2d = dOut.reshape(-1, dOut.shape[-1])  # (N, S)
        N, S = dOut_2d.shape

        # out_2d (N, S) - результат forward'а
        # Построим Якоби-матрицу размера (N, S, S).
        # При обычном 1D-softmax: d p_i / d z_j = p_i (delta_{ij} - p_j).
        # Можно сделать через einsum, как у вас, или циклом:
        # grads = np.einsum(...)  <-- обычно слишком громоздко. 

        # Самый наглядный метод: для каждого из N элементов создаём матрицу (S,S).
        # Но (N, S, S) может быть ОЧЕНЬ большим, если S велик. 
        # Поэтому часто делают батчевую матричную операцию без явного хранения.

        # Поясним здесь вариант с "вектор-множителем":

        # dp/dz = p * (I - p^T), где p - row vector shape (S,), 
        # I - (S,S), p^T - (S,1), но для каждой строки отдельно.

        # 1) dX_2d[i] = dOut_2d[i] @ Jacobi_i
        #    Jacobi_i[j, k] = out_2d[i,j]*(delta_{jk} - out_2d[i,k]) 
        # 2) Экономим память: dX_2d[i] = (dOut_2d[i] - sum(dOut_2d[i])*out_2d[i]) * out_2d[i]
        #    Это известная формула, эквивалентная умножению на Якоби 1D-Softmax.
        #    См. ссылку: https://deepnotes.io/softmax-backprop

        # Реализуем этот компактный вариант:
        # dX = p * (dOut - (dOut·p))   (покомпонентно)
        
        # Скалярное произведение dOut·p (по оси=1):
        dot = np.sum(dOut_2d * self.out_2d, axis=1, keepdims=True)  # (N, 1)

        # Вычитаем из dOut_2d => (dOut_2d - dot * out_2d)
        tmp = dOut_2d - dot * self.out_2d  # (N, S)

        # И умножаем покомпонентно на out_2d
        dX_2d = tmp * self.out_2d  # (N, S)

        # Возвращаем форму (B, H, S, S)
        return dX_2d.reshape(self.original_shape)

    


