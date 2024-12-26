import numpy as np

class Initializer:
    def __init__(self):
        pass

class GlorotUniform(Initializer):
    def __call__(self, shape:tuple, rnd_state:int=42):
        rng = np.random.default_rng(rnd_state)
        limit = np.sqrt(6 / np.sum(shape))
        return rng.uniform(low=-limit, high=limit, size=shape)   

class Zeros(Initializer):
    def __call__(self, shape, rnd_state=None):
        return np.zeros(shape=shape)