import jax
import jax.numpy as jnp

from functools import partial

class TrajectorySampler:
    def __init__(self, sigma=None):
        """Initialize sampler with optional noise scale sigma."""
        # self.sigma = sigma
        # self.is_stochastic = sigma is not None
        # these are not used
        pass

    @partial(jax.jit, static_argnums=(0,1,2))
    def sample_trajectory(self, vf_fn, diffusion_fn, x0, ts, num_steps, key):
        """Sample a trajectory using the vector field function."""
        init_carry = (x0, ts[0], key)

        # Create is_last flags
        is_last_flags = jnp.array([False] * (len(ts) - 2) + [True])

        # Now scan over (t, is_last)
        scan_inputs = (ts[1:], is_last_flags)

        final_carry, traj = jax.lax.scan(
            lambda carry, t_and_flag: self._step(vf_fn, diffusion_fn, carry, *t_and_flag),
            init_carry,
            scan_inputs
        )

        x1 = final_carry[0]
        traj = jnp.moveaxis(traj, 0, 1)
        traj = jnp.concatenate([jnp.expand_dims(x0, 1), traj], axis=1)
        return traj, x1

    
    def _step(self, vf_fn, diffusion_fn, carry, t):
        """Base step method to be implemented by subclasses."""
        raise NotImplementedError

class EulerSampler(TrajectorySampler):

    def _step(self, vf_fn, diffusion_fn, carry, t, is_last):
        x, prev_t, key = carry
        dt = t - prev_t
        ts = prev_t * jnp.ones((x.shape[0],))
        v = vf_fn(x, ts)
        sigma = diffusion_fn(x, ts)

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape)

        # Conditionally apply noise
        noise_term = jnp.where(is_last, 0.0, sigma * jnp.sqrt(jnp.abs(dt)) * noise)
        x_next = x + v * dt + noise_term

        return (x_next, t, key), x_next