import numpy as np

class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Соберем все параметры слоев
        self.params = []
        for layer in layers:
            for name, param in layer.get_params().items():
                self.params.append((layer, name, param))
        
        # Инициализируем моменты для каждого параметра
        self.m = { (id(layer), name): np.zeros_like(param) for layer, name, param in self.params }
        self.v = { (id(layer), name): np.zeros_like(param) for layer, name, param in self.params }
        
        self.t = 0  # Счетчик шагов

    def step(self):
        self.t += 1
        for layer, name, param in self.params:
            grads = layer.get_grads()  # словарь с именами параметров и градиентами
            dparam = grads[name]  # градиент для конкретного параметра
            
            key = (id(layer), name)
            
            # Обновляем моменты
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * dparam
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (dparam ** 2)
            
            # Коррекция на смещение
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Обновляем параметры
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
