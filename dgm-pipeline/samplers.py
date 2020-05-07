import tensorflow as tf
import tensorflow_probability as tfp

from warnings import warn
from time     import time

def init_kernel(target_log_prob_fn, bijectors, method='nuts', step_size=0.05, num_adaptation=500):
    
    if method == 'nuts':
        inner_kernel=tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn,
            step_size=step_size,max_tree_depth=10)
        accept_prob = 0.8

        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.target_log_prob,
                pkr.inner_results.inner_results.leapfrogs_taken,
                pkr.inner_results.inner_results.has_divergence,
                pkr.inner_results.inner_results.energy,
                pkr.inner_results.inner_results.log_accept_ratio
                )

    elif method == 'mala':
        inner_kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn, step_size)
        accept_prob = 0.4

        def trace_fn(_, pkr):
            return None

    kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel, bijector=bijectors)

    wrapped_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        target_accept_prob=accept_prob,
        num_adaptation_steps=num_adaptation,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio
    )
    return wrapped_kernel, trace_fn

def init_remc_kernel(target_log_prob_fn, n_chains, bijectors, leapfrog_steps=4):
    inverse_temps = 0.5 ** tf.range(n_chains)
    step_size = 0.5 / tf.sqrt(inverse_temperatures)

    def make_kernel_fn(target_log_prob_fn, seed):
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            seed=seed, step_size=step_size, num_leapfrog_steps=3)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=target.log_prob,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn)

    def trace_fn(unused_state, results):
        return (results.is_swap_proposed_adjacent,
                results.is_swap_accepted_adjacent)

    return remc, trace_fn


def run_mcmc(init_state, kernel, trace_fn,
              num_steps=500, burnin=500, use_xla=True):
    '''
    Applies MCMC using a desired kernel function.

    Arguments
    ---------
    init_state : list of Numpy arrays
        Starting point for the Markov chain
    kernel : TFP TransitionKernel
        MCMC kernel function for generating proposals
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
        Diagnostics for MCMC performance
    '''

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
      kernel=kernel,
      trace_fn=trace_fn)
    total = int(time() - start)
    warn(f'{total} seconds elapsed during.',UserWarning)

    return [x.numpy() for x in chain_state] , sampler_stat

def ais_loglike(target_logp, proposal_logp,n_chains,input_shape,
                num_ais_steps=1000, hmc_step_size=0.1, 
                hmc_leapfrog_steps=2):
    '''
    Runs annealed importance sampling using Hamiltonian Monte Carlo
    to estimate the importance weights for a ratio of partition functions.
    '''

    weight_samples, ais_weights, kernel_results = (
        tfp.mcmc.sample_annealed_importance_chain(
          num_steps=num_ais_steps,
          proposal_log_prob_fn=proposal_logp,
          target_log_prob_fn=target_logp,
          current_state=tf.zeros([n_chains] + input_shape),
          make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=tlp_fn,
            step_size=hmc_step_size,
            num_leapfrog_steps=hmc_leapfrog_steps)))

    return weight_samples, ais_weights, kernel_results
