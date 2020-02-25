import tensorflow as tf

from functools import partial
from utils     import parse_spec_optimizer, parse_spec_prior
from tqdm      import tqdm

from tensorflow.keras.losses import BinaryCrossentropy

valid_priors = ['normal','uniform']
valid_loss = ['cross_entropy']

class OptimizeIndependent:

    def __init__(self,data,generative_net,inpaint_spec):

        self.latent_dim = generative_net.input.shape[1]
        self.data = data
        if data.shape[0] > 1:
            raise ValueError('Only single images can be inpainted with this method.')

        self.spec = inpaint_spec

        from_logits = inpaint_spec['logits_generated']
        self.n_samples = inpaint_spec['n_samples']

        if inpaint_spec['loss'] == 'cross_entropy':
            self.loss_fn = BinaryCrossentropy(from_logits=from_logits)
        else:
            raise ValueError('The provided loss type is not currently \
            supported. Please try one of {0}.'.format(valid_loss))

        self.prior = parse_spec_prior(self.spec, self.n_samples, self.latent_dim)
        self.opt = parse_spec_optimizer(self.spec)

        self.generative_net = generative_net

    def draw(self,spec,return_latent=False):

        iters = spec['iterations']
        noise_sd = spec['noise_sd']**0.5
        noise_decay = spec['noise_decay']

        z = tf.Variable(self.prior())
        tiled_data = tf.repeat(self.data, self.n_samples, axis=0)
        tiled_mask = tf.repeat(self.data.mask, self.n_samples, axis=0)
        for i in tqdm(range(iters)):
            with tf.GradientTape() as tape:
                x_pred = self.generative_net(z)
                loss = self.loss_fn(tiled_data,x_pred,sample_weight=tiled_mask)

            grad_z = tape.gradient(loss,z)
            self.opt.apply_gradients(zip([grad_z],[z]))

            z = inject_noise(z,noise_sd)
            noise_sd *= noise_decay

        if return_latent:
            return z
        else:
            return self.generative_net(z)


def inject_noise(variable,noise_sd,dtype='float32'):
    '''Add isotropic Gaussian noise to Tensorflow Variable'''
    noise = tf.random.normal(shape=variable.shape,stddev=noise_sd,dtype='float32')
    variable.assign_add(noise)
    return variable
