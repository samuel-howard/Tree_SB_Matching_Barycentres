import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Collection

def get_timestep_embedding(timestep, embedding_dim: int):
    """
    Compute sinusoidal embeddings for a single timestep.
    """
    half_dim = embedding_dim // 2
    k = 10000
    emb = jnp.log(k) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
    return emb

class MLP(nn.Module):
    dim_hidden: Sequence[int] = (128, 128, 128)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = self.activation(Wx(z))
        z = nn.Dense(self.out_dim, use_bias=True)(z)
        return z

class ScoreMLP(nn.Module):
    dim_hidden: Sequence[int] = (128, 128, 128)
    emb_dim_hidden: Sequence[int] = (64, 64)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # x: (B, feature_dim)
        # t: (B,)

        # out: (B, out_dim)
        
        x_emb = MLP(
            dim_hidden = self.emb_dim_hidden,
            activation=self.activation,
            out_dim = -(self.dim_hidden[0] // -2)
        )(x)  # (B, emb_dim)

        t = t[:,None]   # (B, 1)

        t_emb = jax.vmap(partial(get_timestep_embedding, embedding_dim=64))(t) # (B, emb_dim)
        t_emb = MLP(
            dim_hidden = self.emb_dim_hidden,
            activation=self.activation,
            out_dim = self.dim_hidden[0] // 2
        )(t_emb)  # (B, emb_dim)
        
        # Concatenate x_emb and t_emb
        vec = jnp.concatenate([x_emb, t_emb], axis=-1)  # (B, total_emb_dim)
        
        out = MLP(
            dim_hidden = self.dim_hidden,
            activation=self.activation,
            out_dim = self.out_dim
        )(vec)  # (B, out_dim)
        
        return out #.squeeze(-1)  # (B,)


class BasicModel(nn.Module):
    '''
    Implements the basic model from Noble et al. (2023), De Bortoli et al. (2021)
    '''
    # dim_hidden: Sequence[int] = (128, 128, 128)
    # emb_dim_hidden: Sequence[int] = (64, 64)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    out_dim: int = 1
    d: int=2

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # x: (B, feature_dim)
        # t: (B,)

        # out: (B, out_dim)
        
        x_emb = MLP(
            dim_hidden = (128,), #self.emb_dim_hidden,
            activation=self.activation,
            out_dim = max(256, 2*self.d)
        )(x)  # (B, emb_dim)

        t = t[:,None]   # (B, 1)

        t_emb = jax.vmap(partial(get_timestep_embedding, embedding_dim=32))(t) # (B, emb_dim)
        t_emb = MLP(
            dim_hidden = (128,), #self.emb_dim_hidden,
            activation=self.activation,
            out_dim = max(256, 2*self.d) #self.dim_hidden[0] // 2
        )(t_emb)  # (B, emb_dim)
        
        # Concatenate x_emb and t_emb
        vec = jnp.concatenate([x_emb, t_emb], axis=-1)  # (B, total_emb_dim)
        
        out = MLP(
            dim_hidden = (max(256, 2*self.d), max(128, 2*self.d)), #self.dim_hidden,
            activation=self.activation,
            out_dim = self.out_dim
        )(vec)  # (B, out_dim)
        
        return out #.squeeze(-1)  # (B,)