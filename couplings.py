import jax
import jax.numpy as jnp

class Coupling:
    def sample(self, key, flow, num_samples):
        raise NotImplementedError
    
class IndependentCoupling(Coupling):
    def __init__(self):
        pass

    def sample(self, key, flow, num_samples):
        source_key, target_key = jax.random.split(key)
        x0 = flow.source.sample(source_key, num_samples)
        x1 = flow.target.sample(target_key, num_samples)
        return x0, x1