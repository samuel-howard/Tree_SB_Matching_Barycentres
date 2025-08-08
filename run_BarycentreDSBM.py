## implementation of TreeDSBM for the barycentre setting

import jax
import jax.numpy as jnp

import optax

from functools import partial

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
from flax import struct

from tqdm import trange

from couplings import Coupling, IndependentCoupling
from interpolants import BrownianBridge
from samplers import EulerSampler


@flax.struct.dataclass
class TrainState:
    step: int
    edge_sigma: float
    params: Any
    ema_params: Any
    opt_state: Any
    lr: float
    ema_rate: float=0.01


def make_step_fn(flow_loss_fn, model, optimizer):
    '''
    returns step function for training
    '''

    def loss_fn(params, key, batch):
        loss = flow_loss_fn(params, key, *batch)
        return loss
    
    def train_step_fn(key, state, batch):
        # batch = (x0, x1, ts)
        loss, grads = jax.value_and_grad(loss_fn)(state.params, key, batch)
        updates, opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_ema_params = optax.incremental_update(new_params, state.ema_params, state.ema_rate) 
        step = state.step + 1

        new_state = state.replace(
            step=step,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=opt_state,
        )
        return new_state, loss

    return train_step_fn

class t_Dist:
    def sample(self, key, num_samples):
        raise NotImplementedError

class UniformDist(t_Dist):
    def sample(self, key, num_samples):
        return jax.random.uniform(key, (num_samples,), minval=0.001, maxval=1.0-0.001)
    

class Flow:
    '''
    Base class for a Flow object
    Flow objects contain the flow problem
    - source and target distributions
    - t distribution
    - coupling (Independent, minibatch OT etc)
    - interpolant (Linear, Slerp etc)
    - sampler (Euler, Heun, RK4 etc)

    Functionality:
    - sample a batch of data (x0, x1, t), according to the coupling
    '''
    def __init__(
            self,
            shape,
            source,
            target,
            coupling,
            interpolant,
            t_dist,
            sampler,
            sigma=0.0
    ):
        self.shape = shape
        self.source = source
        self.target = target
        self.coupling = coupling
        self.interpolant = interpolant
        self.t_dist = t_dist
        self.sampler = sampler
        self.sigma = sigma

    # @partial(jax.jit, static_argnums=(0,2))
    def sample_batch(self, key, batch_size):
        '''
        Sample a batch of data (x0, x1, t)
        '''
        samples_key, t_key = jax.random.split(key, 2)
        x0, x1 = self.coupling.sample(samples_key, self, batch_size)    # Can change for other couplings, e.g. IMF or minibatch Sinkhorn
        ts = self.t_dist.sample(t_key, batch_size)
        # xt = jax.vmap(self.interpolant.interpolate)(x0, x1, ts)
        batch = (x0, x1, ts)
        return batch
    


class BridgeMatching:
    '''
    Base class for a BridgeMatching object
    BridgeMatching objects contain the vector field model and training procedure
    Takes a flow object at initialisation
    '''
    def __init__(
            self,
            shape,
            flow,
            model,
    ):
        self.shape = shape
        self.flow = flow
        self.model = model
        self.params = None
        self.diffusion_fn = self.get_diffusion_fn()

    def loss_fn(self, params, key, x0, x1, ts):
        '''
        Loss function for the flow. Optimises in both directions, as in De Bortoli et al. 2024.
        '''
        fwd_params, bwd_params = params
        keys = jax.random.split(key, x0.shape[0])
        xt = jax.vmap(self.flow.interpolant.interpolate)(x0, x1, ts, keys)

        # fwd loss
        vt = jax.vmap(self.flow.interpolant.velocity)(x0, x1, xt, ts)
        vt_pred = self.model.apply(fwd_params, xt, ts)
        fwd_loss = jnp.sum((vt_pred - vt) ** 2) / x0.shape[0]

        # bwd loss
        vt = jax.vmap(self.flow.interpolant.velocity)(x1, x0, xt, 1-ts)
        vt_pred = self.model.apply(bwd_params, xt, 1-ts)
        bwd_loss = jnp.sum((vt_pred - vt) ** 2) / x0.shape[0]

        return 0.5 * (fwd_loss + bwd_loss)
    

    def train(self, key, num_training_steps, batch_size=64, lr=1e-3, ema_rate=0.01):

        # initialise the optimisation procedure
        single_init_params = self.model.init(key, jnp.ones((1,*self.shape)), jnp.ones((1,)))
        init_params = (single_init_params, single_init_params)
        optimizer = optax.adam(learning_rate=lr)
        state = TrainState(step=0,
                           params=init_params,
                           ema_params=init_params,
                           opt_state=optimizer.init(init_params),
                           lr=lr,
                           ema_rate=ema_rate,
                           apply_fn=self.model.apply
                           )
        
        train_step_fn = make_step_fn(self.loss_fn, self.model, optimizer)
        train_step_fn = jax.jit(train_step_fn)

        # training
        losses = []
        eval_freq = 10
        with trange(num_training_steps, desc="Training", unit="step") as pbar:
            for step in pbar:
                key, batch_subkey, train_subkey = jax.random.split(key, 3)
                batch = self.flow.sample_batch(batch_subkey, batch_size)
                state, loss = train_step_fn(train_subkey, state, batch)
                losses.append(loss)

                if step % eval_freq == 0:
                    pbar.set_postfix(loss=loss.item())

        return state, losses
    
    def get_drift_fn(self, state, use_ema_params, fwd=True):
        '''
        Get the velocity field function
        '''
        if use_ema_params:
            drift_params = state.ema_params
        else:
            drift_params = state.params

        if fwd:
            drift_params = drift_params[0]
        else:
            drift_params = drift_params[1]

        def drift_fn(x, t):
            return self.model.apply(drift_params, x, t)
        return drift_fn
    
    def get_diffusion_fn(self):
        '''
        Get the diffusion field function
        '''
        def diffusion_fn(x, t):
            return self.flow.sigma
        return diffusion_fn
    
    def sample(self, key, drift_fn, num_samples, num_steps=100, fwd=True, x0=None):
        '''
        Sample a trajectory according to an input velocity field
        Optionally, start at x0
        '''
        if x0 is None:
            if fwd:
                x0 = self.flow.source.sample(key, num_samples)
            else:
                x0 = self.flow.target.sample(key, num_samples)

        ts = jnp.linspace(0, 1, num_steps+1) # Uniform schedule

        traj, x1 = self.flow.sampler.sample_trajectory(drift_fn, self.diffusion_fn, x0, ts, num_steps, key)
        return traj, x1
    

## Couplings:

class MultiIndependentCoupling(Coupling):
    '''
    Used for training the initial IMF iteration.
    Sample independently from the independent coupling over the marginals.
    Construct the barycentre sample according to samples from this coupling.
    '''
    def __init__(self, mu_lst, weights, shape, idx, sigma):
        self.mu_lst = mu_lst
        self.weights = weights
        self.shape = shape
        self.idx = idx
        self.sigma = sigma

    @partial(jax.jit, static_argnums=(0,2,3))
    def sample(self, key, flow, num_samples):
        mu_key, z_key = jax.random.split(key)

        mu_keys = jax.random.split(mu_key, len(self.mu_lst))
        samples = jnp.stack([mu.sample(k, num_samples) for mu, k in zip(self.mu_lst, mu_keys)], axis=0)
        nu_samples = jnp.einsum('n...,n->...', samples, self.weights)
        x0 = samples[self.idx]
        
        noise = jax.random.normal(z_key, (num_samples, *self.shape)) * self.sigma
        x1 = nu_samples + noise
        return x0, x1
    

class BranchCoupling(Coupling):
    '''
    Coupling that uses fixed samples, along a single edge.
    At initialisation, takes in two sets of samples (x0, x1) from the current coupling (but only one edge).
    Returns random samples from the two sets, with added noise to the barycentre sample
    '''
    def __init__(self, x0, x1, sigma):
        self.x0s = x0
        self.x1s = x1
        self.total_batch_size = x0.shape[0]
        self.sigma = sigma

    def sample(self, key, flow, num_samples):
        idxs = jax.random.randint(key, (num_samples,), 0, self.total_batch_size)
        zs = jax.random.normal(key, (num_samples,) + self.x0s.shape[1:]) * self.sigma # the original sigma - not the one in the flow
        return self.x0s[idxs], self.x1s[idxs] + zs
    


class BarycentreDSBM:
    '''
    Class for the TreeDSBM algorithm, for the barycentre setting.
    '''
    def __init__(
            self,
            mu_lst,
            sigma,
            shape,
            model,
            weights=None,
    ):
        self.mu_lst = mu_lst
        self.sigma = sigma
        self.shape = shape
        self.K = len(mu_lst)
        self.model = model
        self.weights = weights if weights is not None else jnp.ones(self.K) / self.K

    def interpolate(self, x0, x1, t, key, edge_sigma):
        sigma = jnp.sqrt(t * (1 - t)) * edge_sigma
        noise = jax.random.normal(key, x0.shape)
        return (1 - t) * x0 + t * x1 + sigma * noise

    def velocity(self, x0, x1, xt, t):
        return (x1 - xt) / (1 - t + 1e-8)

    def bm_loss_fn(self, params, key, state, x0, x1, ts):
        '''
        Loss function for the flow. Optimises in both directions, as in De Bortoli et al. 2024.
        '''
        fwd_params, bwd_params = params
        keys = jax.random.split(key, x0.shape[0])
        xt = jax.vmap(self.interpolate, in_axes=(0, 0, 0, 0, None))(x0, x1, ts, keys, state.edge_sigma)

        # fwd loss
        vt = jax.vmap(self.velocity)(x0, x1, xt, ts)
        vt_pred = self.model.apply(fwd_params, xt, ts)
        fwd_loss = jnp.sum((vt_pred - vt) ** 2) / x0.shape[0]

        # bwd loss
        vt = jax.vmap(self.velocity)(x1, x0, xt, 1-ts)
        vt_pred = self.model.apply(bwd_params, xt, 1-ts)
        bwd_loss = jnp.sum((vt_pred - vt) ** 2) / x0.shape[0]

        return 0.5 * (fwd_loss + bwd_loss)

    @partial(jax.jit, static_argnums=(0,4))
    def bm_train_step(self, key, state, batch, train_config):
        '''
        vmappable training step
        - should do batches
        - and update the state amd stuff
        '''
        # train step
        loss, grads = jax.value_and_grad(self.bm_loss_fn)(state.params, key, state, *batch)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_ema_params = optax.incremental_update(new_params, state.ema_params, state.ema_rate) 
        step = state.step + 1

        new_state = state.replace(
            step=step,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=opt_state,
        )

        return new_state, loss
    
    def bm_train_single(self, key, state, bm, train_config, is_first_IMF):
        '''
        Train a single BridgeMatching object.
        '''

        losses = []
        eval_freq = 10

        if train_config.reflow_num_training_steps is not None:
            if is_first_IMF:
                num_training_steps = train_config.num_training_steps
            else:
                num_training_steps = train_config.reflow_num_training_steps
        else:
            num_training_steps = train_config.num_training_steps

        with trange(num_training_steps, desc="Training", unit="step") as pbar:
            for step in pbar:
                key, batch_subkey, train_subkey = jax.random.split(key, 3)

                batch = bm.flow.sample_batch(batch_subkey, train_config.batch_size)

                state, loss = self.bm_train_step(train_subkey, state, batch, train_config)

                if step % eval_freq == 0:
                    pbar.set_postfix(loss=loss.item())

        return state, losses

    def bm_train_sequential(self, key, states_lst, bm_lst, train_config, is_first_IMF=False):
        '''
        Trains the BridgeMatching object along each edge sequentially.
        '''
        new_states_lst = []
        losses_lst = []

        for i in range(self.K):
            key, subkey = jax.random.split(key, 2)

            new_state, losses = self.bm_train_single(subkey, states_lst[i], bm_lst[i], train_config, is_first_IMF)

            new_states_lst.append(new_state)
            losses_lst.append(losses)

        return new_states_lst, losses_lst

    # functions for simultaneous training (on single GPU)
    @partial(jax.jit, static_argnums=(0,2))
    def get_parallel_initial_batches(self, key, train_config):
        '''
        Get batches for the first IMF step.
        Returns batches, with K as leading dimension.
        '''
        mu_key, z_key = jax.random.split(key)

        mu_keys = jax.random.split(mu_key, len(self.mu_lst))
        samples = jnp.stack([mu.sample(k, train_config.batch_size) for mu, k in zip(self.mu_lst, mu_keys)], axis=0)

        nu_samples = jnp.einsum('n...,n->...', samples, self.weights)
        noise = jax.random.normal(z_key, (train_config.batch_size, *self.shape)) * self.sigma
        nu_samples = nu_samples + noise

        t_keys = jax.random.split(key, self.K)
        ts = jax.vmap(lambda k: jax.random.uniform(k, (train_config.batch_size,), minval=0.001, maxval=1.0-0.001))(t_keys)

        batches = [ (samples[i], nu_samples, ts[i]) for i in range(self.K) ]

        stacked_batches = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *batches)

        return stacked_batches

    def get_parallel_reflow_batches_single(self, key, coupling_samples, train_config, edge_idx):
        '''
        Get a single batch for the reflow step, from the coupling samples.
        '''
        idxs_key, t_key = jax.random.split(key, 2)

        total_batch_size = coupling_samples.shape[1]
        idxs = jax.random.randint(idxs_key, (train_config.batch_size,), 0, total_batch_size)

        zs = jax.random.normal(key, (train_config.batch_size,) + self.shape) * self.sigma  # the original sigma - not the one in the flow

        x0 = coupling_samples[edge_idx][idxs]
        x1 = coupling_samples[-1][idxs] + zs
        ts = jax.random.uniform(t_key, (train_config.batch_size,), minval=0.001, maxval=1.0-0.001)

        return x0, x1, ts

    @partial(jax.jit, static_argnums=(0,3))
    def get_parallel_reflow_batches(self, key, coupling_samples, train_config):
        '''
        Vmap over the marginals
        '''
        subkeys = jax.random.split(key, self.K)

        batches = jax.vmap(self.get_parallel_reflow_batches_single, in_axes=(0, None, None, 0))(subkeys, coupling_samples, train_config, jnp.arange(self.K))

        return batches


    def bm_train_parallel(self, key, states_lst, bm_lst, train_config, is_first_IMF=False, coupling_samples=None):
        '''
        Trains the BridgeMatching objects along each edge simultaneously.
        '''
        losses_lst = []
        eval_freq = 10

        # stack TrainStates for vmap
        stacked_states = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *states_lst)

        if train_config.reflow_num_training_steps is not None:
            if is_first_IMF:
                num_training_steps = train_config.num_training_steps
            else:
                num_training_steps = train_config.reflow_num_training_steps
        else:
            num_training_steps = train_config.num_training_steps

        with trange(num_training_steps, desc="Training", unit="step") as pbar:
            for step in pbar:

                # get batches
                key, subkey = jax.random.split(key, 2)
                if is_first_IMF:
                    batches = self.get_parallel_initial_batches(subkey, train_config)
                else:
                    batches = self.get_parallel_reflow_batches(subkey, coupling_samples, train_config)

                # train step
                subkeys = jax.random.split(key, self.K)

                stacked_states, losses = jax.vmap(self.bm_train_step, in_axes=(0, 0, 0, None))(subkeys, stacked_states, batches, train_config)
                losses = jnp.array(losses)

                if step % eval_freq == 0:
                    pbar.set_postfix(loss=jnp.mean(losses).item())

                losses_lst.append(losses)

        # Unstack the states back into a list
        leaves, treedef = jax.tree_util.tree_flatten(stacked_states)
        unstacked_states_list = [
            treedef.unflatten(leaf[i] for leaf in leaves)
            for i in range(self.K)
        ]

        return unstacked_states_list, losses_lst
        
    def get_multi_coupling(self, key, bm_lst, states_lst, num_steps=100, num_samples=4096, start_idx=0, use_ema_params=True, return_trajs=False, batch_size=None):
        '''
        Memory-efficient version: simulates in batches to avoid OOM errors.
        Returns coupling samples, shape (num_marginals+1, num_samples, ...). Final dimension is the barycentre samples.
        Optionally returns the trajectories.
        '''

        num_marginals = len(bm_lst)
        shape = bm_lst[0].shape  # assumes all BMs produce samples of the same shape

        if batch_size is None:
            batch_size = num_samples

        samples = jnp.zeros((num_marginals + 1, num_samples, *shape))
        if return_trajs:
            trajs_lst = jnp.zeros((num_marginals, num_samples, num_steps + 1, *shape))

        # First sample barycenter and start marginal trajectories
        start_bm = bm_lst[start_idx]
        state = states_lst[start_idx]
        drift_fn = start_bm.get_drift_fn(state, use_ema_params, fwd=True)
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_size_actual = batch_end - i
            key, subkey = jax.random.split(key)

            traj, ys = start_bm.sample(subkey, drift_fn, batch_size_actual, num_steps, fwd=True)

            samples = samples.at[start_idx, i:batch_end].set(traj[:, 0])
            samples = samples.at[-1, i:batch_end].set(ys)
            if return_trajs:
                trajs_lst = trajs_lst.at[start_idx, i:batch_end].set(traj)

        # Now for each other marginal, simulate in batches
        for j in range(num_marginals):
            if j == start_idx:
                continue

            bm = bm_lst[j]
            state = states_lst[j]
            drift_fn = bm.get_drift_fn(state, use_ema_params, fwd=False)

            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_size_actual = batch_end - i
                key, subkey = jax.random.split(key)
                ys_batch = samples[-1, i:batch_end]  # use barycenter samples as x0
                traj, xi = bm.sample(subkey, drift_fn, batch_size_actual, num_steps, fwd=False, x0=ys_batch)

                samples = samples.at[j, i:batch_end].set(xi)
                if return_trajs:
                    trajs_lst = trajs_lst.at[j, i:batch_end].set(traj)

        if return_trajs:
            return samples, trajs_lst
        else:
            return samples
        

    def train(
            self,
            key,
            train_config,
            model,
    ):
        '''
        Train the BarycentreDSBM algorithm.
        '''

        self.optimizer = optax.adam(learning_rate=train_config.lr)

        key, init_key = jax.random.split(key, 2)
        single_init_params = model.init(init_key, jnp.ones((1,*self.shape)), jnp.ones((1,)))
        init_params = (single_init_params, single_init_params)

        # Initial Bridge Matching step
        states_lst = []
        bm_lst = []
        flows_lst = []

        # change this
        for i in range(self.K):
            edge_sigma = self.sigma / self.weights[i]
            flow = Flow(
                shape=self.shape,
                source=self.mu_lst[i],
                target=None, # Placeholder: not used
                coupling=MultiIndependentCoupling(self.mu_lst, self.weights, self.shape, i, self.sigma),
                interpolant=BrownianBridge(sigma=edge_sigma),
                t_dist=UniformDist(),
                sampler=EulerSampler(),
                sigma=edge_sigma # equivalent to running for different amount of time
            )
            flows_lst.append(flow)

            bm = BridgeMatching(shape=self.shape, flow=flow, model=model)
            bm_lst.append(bm)

            state = TrainState(step=0,
                           params=init_params,
                           ema_params=init_params,
                           opt_state=self.optimizer.init(init_params),
                           lr=train_config.lr,
                           ema_rate=train_config.ema_rate,
                           edge_sigma=edge_sigma
                           )

            states_lst.append(state)

        # Training Bridge Matching
        print(f"Running IMF step 1")
        key, subkey = jax.random.split(key, 2)
        if train_config.simultaneous_training:
            states_lst, losses_lst = self.bm_train_parallel(subkey, states_lst, bm_lst, train_config, is_first_IMF=True)
        else:
            states_lst, losses_lst = self.bm_train_sequential(subkey, states_lst, bm_lst, train_config, is_first_IMF=True)


        # Subsequent IMF steps

        all_states = [states_lst]
        all_bm_lst = [bm_lst]

        for reflow_step in range(train_config.num_IMF_steps - 1):

            print(f"Running IMF step {reflow_step+2}")

            all_coupling_samples = []
            key, subkey = jax.random.split(key, 2)
            for start_idx in range(self.K):
                subkey_i = jax.random.fold_in(subkey, start_idx)  # ensure different randomness per run
                coupling_samples = self.get_multi_coupling(
                    key=subkey_i,
                    bm_lst=all_bm_lst[-1],
                    states_lst=all_states[-1],
                    num_steps=train_config.num_sampling_steps,
                    num_samples=train_config.num_training_samples // self.K,
                    start_idx=start_idx,
                    batch_size=train_config.simulation_batch_size,
                )
                all_coupling_samples.append(coupling_samples)


            # Stack along a new axis, then reshape to combine samples
            coupling_samples = jnp.concatenate(all_coupling_samples, axis=1)

            new_nu_samples = jnp.einsum('n...,n->...', coupling_samples[:-1, :], self.weights)

            coupling_samples = coupling_samples.at[-1].set(new_nu_samples)

            states_lst = []
            bm_lst = []
            flow_lst = []

            # define the problems along each edge
            for i in range(self.K):
                # use the simulated coupling for training
                coupling = BranchCoupling(coupling_samples[i, :, :], new_nu_samples, self.sigma)
                edge_sigma = self.sigma / self.weights[i]

                flow = Flow(
                    shape=self.shape,
                    source=self.mu_lst[i],
                    target=None, # placeholder - not used
                    coupling=coupling,
                    interpolant=BrownianBridge(sigma=edge_sigma), 
                    t_dist=UniformDist(),
                    sampler=EulerSampler(),
                    sigma=edge_sigma # equivalent to running for different amount of time
                )
                bm = BridgeMatching(shape=self.shape, flow=flow, model=model)
                bm_lst.append(bm)

                if train_config.warmstart:
                    init_params = all_states[0][i].ema_params

                state = TrainState(step=0,
                            params=init_params,
                            ema_params=init_params,
                            opt_state=self.optimizer.init(init_params),
                            lr=train_config.lr,
                            ema_rate=train_config.ema_rate,
                            edge_sigma=edge_sigma
                        )

                states_lst.append(state)


            # Training Bridge Matching
            key, subkey = jax.random.split(key, 2) 

            if train_config.simultaneous_training:
                states_lst, losses_lst = self.bm_train_parallel(
                    subkey, states_lst, bm_lst, train_config, is_first_IMF=False, coupling_samples=coupling_samples
                )
            else:
                states_lst, losses_lst = self.bm_train_sequential(subkey, states_lst, bm_lst, train_config, is_first_IMF=False)

            all_states.append(states_lst)
            all_bm_lst.append(bm_lst)

        # Return states and bm_lst for all IMF steps
        return all_states, all_bm_lst