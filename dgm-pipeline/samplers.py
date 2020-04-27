import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from functools import partial
from utils     import parse_spec_optimizer, parse_spec_prior, parse_spec_loss
from tqdm      import trange
from warnings import warn
from time     import time

VERY_LARGE_ENERGY = 1e10
DTYPE = 'float32'

valid_priors = ['normal', 'uniform']

def run_nuts_chain(init_state, step_size, target_log_prob_fn, unconstraining_bijectors,
              num_steps=500, burnin=500, use_xla=True):
    '''
    Applies MCMC using the No-U-Turn Sampler with dual step size adaptation.

    Arguments
    ---------
    init_state : list of Numpy arrays
        Starting point for the Markov chain
    step_size : float
        Size of the steps to be taken by the leapfrog integrator
    target_log_prob_fn : function
        Tensorflow function which yields the log posterior density
    unconstraining_bijectors : list of Bijectors
        List of bijector functions used to map the parameters from
        their original scale to an unconstrained scale
    num_steps : int
        Number of samples drawn after burn-in
    burnin : int
        Number of iterations to be discarded before recording samples
    use_xla : bool
        Determines whether to use XLA compiler for more efficient compute.

    Returns
    -------
    chain_state : list of Tensorflow arrays
        Trace with samples from Markov chain
    sampler_stat : list of Tensorflow arrays
        Diagnostics for NUTS performance
    '''

    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio
               )
  
    kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.NoUTurnSampler(
      target_log_prob_fn,
      step_size=step_size,max_tree_depth=10),
    bijector=unconstraining_bijectors)

    hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=burnin,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio
      )

    # To prevent retracing that occurs when the larger function
    # is supplied non-Tensorflow arguments
    @tf.function(autograph=True, experimental_compile=use_xla)
    def autograph_sample_chain(*args,**kwargs):
        return tfp.mcmc.sample_chain(*args, **kwargs)
    start = time()
    chain_state, sampler_stat = autograph_sample_chain(
      num_results=num_steps,
      num_burnin_steps=burnin,
      current_state=init_state,
      kernel=hmc,
      trace_fn=trace_fn)
    total = int(time() - start)
    warn(r'{total} seconds elapsed during.',UserWarning)
    
    n_diverging = sampler_stat[2]
    
    if n_diverging > 0:
        warn(r'{n_divergences} occurred during sampling.',UserWarning)
    
    return chain_state, sampler_stat
