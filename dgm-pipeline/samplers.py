import tensorflow as tf
import numpy as np

from functools import partial
from utils     import parse_spec_optimizer, parse_spec_prior, parse_spec_loss
from tqdm      import trange

VERY_LARGE_ENERGY = 1e10
DTYPE = 'float32'

valid_priors = ['normal', 'uniform']

class LangevinDynamics:
    '''Class for inpainting algorithm using gradient descent on randomly
    sampled latent states. This is equivalent to Langevin dynamics when
    a nonzero noise level is used.

    Arguments
    ---------
    data : Numpy masked array
        4D array with dimensions [1, height, width, channels] containing
        the image which should be inpainted
    generative_net : Tensorflow Model
        Generative model used to map latent codes into images
    inpaint_spec : dict
        Dictionary of parameter configurations for this algorithm
    '''


    def __init__(self,data,generative_net,inpaint_spec):

        self.latent_dim = generative_net.input.shape[1]
        self.data = data
        self.spec = inpaint_spec

        from_logits = inpaint_spec['logits_generated']
        self.n_samples = inpaint_spec['n_samples']

        # The loss function is not a scalar but instead
        # a vector with a distinct loss for each sample
        no_reduce = tf.keras.losses.Reduction.NONE
        loss_type = inpaint_spec['loss'] 
        self.loss_fn = parse_spec_loss(loss_type, kwargs=inpaint_spec)

        self.opt = parse_spec_optimizer(self.spec)

        self.generative_net = generative_net

    def draw(self,spec,return_latent=False,loss_update=1):
        '''Runs inpainting algorithm to generate image completions.

        Arguments
        ---------
        spec : dict
            Dictionary of sampling parameters
        return_latent : bool
            Determines whether the sampled values should be returned as
            their latent codes or as the decoded images.

        Returns
        -------
        samples : Numpy array
            Sampled completions of the inpainted image.
        '''

        iters = spec['iterations']

        noise_sd    = tf.constant(spec['noise_sd'])
        noise_decay = tf.constant(spec['noise_decay'])

        use_metropolis = spec['use_metropolis']
        skip_grad      = spec['skip_grad']
        apply_tiling   = spec['apply_tiling']

        n_images = self.data.shape[0]
        # The Numpy MaskedArray convention is to assign missing values
        # a mask value of 1 so we invert the mask to only 
        # use computation on the observed portion.
        is_observed = tf.cast(1-self.data.mask,DTYPE)
        if apply_tiling:
            tiled_data = tf.cast(tf.repeat(self.data, self.n_samples, axis=0),DTYPE)
            tiled_mask = tf.repeat(is_observed, self.n_samples, axis=0)
        else:
            tiled_data = tf.cast(self.data, DTYPE)
            tiled_mask = is_observed
 
        history = []
        loss_history = np.zeros(iters)

        n_chains = n_images * self.n_samples
        self.prior = parse_spec_prior(self.spec, n_chains, self.latent_dim)

        # Apply Langevin dynamics with decaying noise
        self.z = tf.Variable(self.prior())
        z_old = tf.Variable(self.z)

        x_init      = self.generative_net(self.z)
        loss_init   = self.loss_fn(tiled_data,x_init) * tiled_mask    
        energy_init = loss_to_1_dim(loss_init)

        energy      = tf.Variable(energy_init)
        energy_old  = tf.Variable(energy_init)

        t = trange(iters,desc='Loss %')

        for i in t:
        
            if use_metropolis:
                loss = mala_step(self.z,z_old,energy_old,self.generative_net,self.loss_fn,self.opt,
                        tiled_mask,tiled_data,noise_sd,energy,skip_grad=skip_grad)
  
            else:
                loss = langevin_step(self.z,tiled_mask,self.generative_net,self.loss_fn,self.opt,
                        tiled_data,noise_sd,skip_grad=skip_grad)
            
            t.set_description(f'Loss: {loss:.4f}')
            noise_sd *= noise_decay

            if spec['return_history'] and (i % spec['save_interval']==0) :
                history.append(self.generative_net(self.z).numpy())

            loss_history[i] = loss

        if spec['return_history']:
            samples = np.stack(history)
        elif return_latent:
            samples = self.z
        else:
            samples = self.generative_net(self.z)

        if spec['from_logits'] and not return_latent:
            samples = tf.sigmoid(samples).numpy()

        return samples, loss_history

@tf.function
def gd_step(z,generative_net,loss_fn,opt,
            mask,data,noise_sd,skip_grad=False):
    '''Single step of gradient descent with masked loss function.'''
    with tf.GradientTape() as tape:
        x_pred = generative_net(z)      
        energy_pixel = loss_fn(data,x_pred) * mask
        energy_pred = loss_to_1_dim(energy_pixel)
    avg_energy = tf.reduce_mean(energy_pred)

    # For some reason the Jacobian usually comes out 
    # with shape [n_samples,1,latent_dim] so 
    # we drop the middle axis.
    if not skip_grad:
        z_jac = tape.batch_jacobian(energy_pred,z)[:,0,:]
        opt.apply_gradients(zip([z_jac],[z]))
    return avg_energy
    
@tf.function   
def langevin_step(z,mask,generative_net,loss_fn,opt,
                data,noise_sd,skip_grad=False):
    '''Single step of Langevin dynamics by adding isotropic noise
    to gradient descent.'''
    loss = gd_step(z,generative_net,loss_fn,opt,
            mask,data,noise_sd,skip_grad=skip_grad)
    inject_noise(z,noise_sd)
    return loss

@tf.function
def mala_step(z,z_old,energy_old,generative_net,loss_fn,opt,
            mask,data,noise_sd,energy,skip_grad=False):
    '''Single step of Metropolis-adjusted Langevin algorithm'''
    z_old.assign(z)
    energy_old.assign(energy)
    loss = langevin_step(z,mask,generative_net,loss_fn,opt,
                data,noise_sd,skip_grad=False)
    x_pred = generative_net(z)
    energy_new_pixel = loss_fn(data,x_pred) * mask
    energy_new = loss_to_1_dim(energy_new_pixel)
    is_accepted = vector_metropolis(energy,energy_new)
    z.assign(is_accepted * z + (1-is_accepted) * z_old )
    energy.assign(is_accepted * energy_new + (1-is_accepted) * energy_old)
    return loss

@tf.function
def vector_metropolis(energy_old,energy_new):
    '''Perform a vectorized Metropolis accept/reject step for multiple Markov chains 
    simultaneously. This function assumes that the cross-chain axis is first in the 
    energy function shape.'''
    delta = energy_new - energy_old
    u = tf.random.uniform(shape=energy_old.shape)
    is_accept = tf.math.logical_or(delta>0, tf.math.exp(delta) > u)
    return tf.cast(is_accept,DTYPE)

@tf.function
def inject_noise(variable,noise_sd,dtype=DTYPE):
    '''Add isotropic Gaussian noise to Tensorflow Variable'''
    noise = tf.random.normal(shape=variable.shape,stddev=noise_sd,dtype=DTYPE)
    variable.assign(variable+noise)
    return 

@tf.function
def loss_to_1_dim(loss_pixel,sum_axis=[1,2,3,]):
    '''Takes a 4D pixel loss and reduces it to a 1D vector of losses
    summed over pixels / channels.'''
    reduced_loss = tf.reduce_sum(loss_pixel,axis=sum_axis)
    return tf.expand_dims(reduced_loss,axis=1)

class ParallelHMC:
    def __init__():
        raise NotImplementedError

class SamplerPyMC3:
    def __init__():
        raise NotImplementedError


#class SamplerPyMC3:
#    '''Sampler using PyMC3 interfacing with Tensorflow. Only
#    appropriate for a single chain at a time.'''
 