import tensorflow as tf
import numpy as np
import json
import fire
import glob
import os

import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.backend import binary_crossentropy, abs

valid_loss   = ['cross_entropy', 'mse', 'mae']

def parse_spec_loss(loss_type,kwargs={}):
    if loss_type == 'cross_entropy':
        loss_fn  = partial(binary_crossentropy,from_logits=kwargs['from_logits'])
    elif loss_type == 'mse':
        loss_fn = lambda true, pred: (true - pred) ** 2 
    elif loss_type == 'mae':
        loss_fn = lambda true, pred: abs(true - pred)
    else:
        raise ValueError('The provided loss type is not currently \
        supported. Please try one of {0}.'.format(valid_loss))
    return loss_fn

def eval_function(pred, true, function, sample_axis=0):
    '''Evaluate function on both ground truth and samples
    of predicted values.'''
    batch = np.swapaxes(pred,0,sample_axis)
    evals_pred = [function(sample) for sample in batch]
    eval_true = function(sample)
    return evals_pred, eval_true

def is_covered(pred, true, width=90):
    high, low = np.percentile(pred,q=[width, 1-width])
    return np.logical_and(high > true, true > low)

def switch_bn_mode(model):
    '''Toggle training behavior for batch norm layers in a Model.'''
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = not layer.trainable
    return model

def parse_spec_optimizer(spec, valid_opt=['adam','sgd']):
    '''Create Keras optimizer object from JSON specification.'''
    opt = spec['optimizer']
    if opt == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'])
    elif opt =='sgd':
        return tf.keras.optimizers.SGD(learning_rate=spec['learning_rate'])
    elif opt not in valid_opt:
        raise ValueError('Optimizer not recognized. Please provide one of {0}.'.format(valid_opt))

def parse_spec_prior(spec, n, latent_dim, valid_prior=['uniform','normal']):
    '''Create prior distribution from JSON specification.'''
    prior = spec['prior_type']
    scale = spec['prior_scale']

    if prior == 'normal':
        return partial(tf.random.normal,shape=[n,latent_dim],stddev=scale)
    elif prior == 'uniform':
        return partial(tf.random.uniform,shape=[n,latent_dim],minval=0,maxval=scale)
    elif prior not in valid_prior:
        raise ValueError('Prior type not recognized. Please provide one of {0}'.format(valid_prior))

def get_wgan_losses_fn():
    '''From github.com/LynnHo GAN repository'''
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss


    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def gradient_penalty(f, real, fake):
    '''Calculate the loss due to gradient penalty under the WGAN-GP model.
    Also from github.com/LynnHo'''
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    gp = _gradient_penalty(f, real, fake)

    return gp

def flatten_image_batch(x,nrows,ncols):
    # Convert 3D array of images into a tiled 2D image
    height,width = x.shape[1:3]
    out = np.empty([nrows*height, ncols*width])
    for i in range(nrows):
        for j in range(ncols):
            out[i*height:(i+1)*height,j*width:(j+1)*width] = x[i*nrows+j]
    return out

def read_and_resize(path,units,order):
    '''Open JSON spec for Keras model and resize
    one of its layers.'''
    with open(path,'r') as src:
        str_spec = src.read()
    str_dict = json.loads(str_spec)
    updated = resize_layer(str_dict,units,order)
    return json.dumps(updated)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def describe_spec(spec):
    #TODO: write function to give brief summary of entire model arch.
    pass
def get_diagnostics(model):
    #TODO: write function to pull out measures of model fit
    pass

def check_json_valid(root_dir='..'):
    json_files = glob.glob(root_dir + '/**/*.json', recursive=True)

    for f in json_files:

        try:
            with open(f,'r') as src:
                json.loads(src.read())

        except json.decoder.JSONDecodeError as error:
            raise Exception('JSON read failed for {0}.'.format(f)) from error

def norm_zero_one(array):
    '''Normalize data to have support between zero and one.
    This function uses the global extrema to normalize.'''
    array = array-array.min()
    array = array / array.max()
    return array

def norm_div_255(array):
    array = array
    return array / 255

def prep_data(array,spec,normalizer=norm_zero_one):
    batch_size = spec['batch_size']
    n = array.shape[0]
    data = normalizer(array)
    raw = tf.data.Dataset.from_tensor_slices(data)
    return raw.shuffle(n).batch(batch_size)

def fix_batch_shape(spec,shape,order=0):
    if isinstance(spec,str):
        spec = json.loads(spec)
    all_layers = spec['config']['layers']
    layer = all_layers[order]['config']
    layer['batch_input_shape'] = shape
    return json.dumps(spec)

def resize_layer(spec,units,order):
    '''Loads in a Tensorflow model architecture and
    resizes one of its layers. The value is returned as a string
    to be interoperable with tf.keras.models.from_json()

    Arguments
    ---------
    spec : dict
        Tensorflow model architecture contained within a dict
    units : number of units or filters that the layer should
        be resized to use
    order : int
        The index of the layer that should be reshaped'''

    if isinstance(spec,str):
        spec = json.loads(spec)

    all_layers = spec['config']['layers']
    layer = all_layers[order]['config']
    for param in ['units','filters']:
        if param in layer.keys():
            layer[param] = units
            return spec

    raise KeyError('Layer {0} could not be updated.'.format(order))


def replace_batch_with_masked(mask_single,image_batch):
    
    assert len(mask_single.shape) == 4
    assert len(image_batch.shape) == 4
    
    batch_size = image_batch.shape[0]
    
    # This array is zero for all pixels that are to be replaced
    # Pixels are replaced if their mask is zero in the original
    # masked array
    is_kept = np.repeat(mask_single.mask,batch_size , axis=0)
    
    # Zero out all the pixels that are NOT masked in the test data
    new_batch = image_batch * is_kept
    
    single_with_zeros = mask_single.data * (1-mask_single.mask)
    repeated_single = np.repeat(single_with_zeros,batch_size,axis=0)
    return new_batch + repeated_single


def batch_kl_diag_mvn(mu1, mu2, var1, var2):
    '''batched KL divergence between two Gaussian distributions with
    diagonal covariance matrix. This assumes the batch dimension comes
    first and the dimensions of the covariance matrices are second.'''

    N = mu1.shape[1]
    logdet1 = tf.math.log(tf.reduce_sum(mu1,axis=1))
    logdet2 = tf.math.log(tf.reduce_sum(mu2,axis=1))
    prec2 = var2**-1
    tr = tf.reduce_sum(prec2 * var1)
    delta_mu_sq = (mu_2 - mu1)**2
    inner_prod_mu = tf.reduce_sum(delta_mu_sq * prec2, axis=1)
    return 0.5 * (logdet2 - logdet1 + tr + inner_prod_mu - N)