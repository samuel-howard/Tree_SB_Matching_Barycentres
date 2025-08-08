import jax
import jax.numpy as jnp

class Interpolant:
    def __init__(self):
        pass

    def interpolate(self, x0, x1, t, key):
       raise NotImplementedError
    
    def velocity(self, x0, x1, xt, t):
        raise NotImplementedError
    
    
class BrownianBridge(Interpolant):

    def __init__(self, sigma=None):
        self.sigma = sigma

    def interpolate(self, x0, x1, t, key):
        sigma = jnp.sqrt(t * (1 - t)) * self.sigma
        noise = jax.random.normal(key, x0.shape)
        return (1 - t) * x0 + t * x1 + sigma * noise

    def velocity(self, x0, x1, xt, t):
        return (x1 - xt) / (1 - t + 1e-8)