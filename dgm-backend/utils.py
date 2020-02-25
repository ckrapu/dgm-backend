import tensorflow as tf
import numpy as np
import json
import fire
import glob

from functools import partial

def parse_spec_optimizer(spec, valid_opt=['adam']):
    opt = spec['optimizer']
    if opt == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'])
    elif opt not in valid_opt:
        raise ValueError('Optimizer not recognized. Please provide one of {0}.'.format(valid_opt))

def parse_spec_prior(spec, n, latent_dim, valid_prior=['uniform','normal']):
    prior = spec['prior_type']
    scale = spec['prior_scale']

    if prior == 'normal':
        return partial(tf.random.normal,shape=[n,latent_dim],stddev=scale)
    elif prior == 'uniform':
        return partial(tf.random.uniform,shape=[n,latent_dim],minval=0,maxval=prior_scale)
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
    '''Also from github.com/LynnHo'''
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

def read_and_resize(path,units,order):
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

def norm_div_255(array):
    return array / 255

def prep_data(array,spec,normalizer=norm_div_255):
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

if __name__ == '__main__':
  fire.Fire()
