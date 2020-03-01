import json
import utils

import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from tqdm      import trange
from os        import remove
from os.path   import exists

SPECS_DIR = '../model_specs/network_specs/'
VIZ_DIR   = '../data/visualizations/'  

class GenerativeModel(tf.keras.Model):
    '''Common class for deep generative models'''
    def __init__(self):
        super(GenerativeModel, self).__init__()

    def save(self, prefix, overwrite=True):
        has_generator = hasattr(self, 'generative_net')
        has_inference = hasattr(self, 'inference_net')

        if has_inference:           
            inference_path = prefix + '_inference_net.h5'
            if exists(inference_path):
                remove(inference_path)
            self.inference_net.save(inference_path, overwrite=overwrite)

        if has_generator:
            generative_path = prefix + '_generative_net.h5'
            if exists(generative_path):
                remove(generative_path)
            self.generative_net.save(generative_path, overwrite=overwrite)

        if not (has_generator or has_inference):
            raise ValueError('No model object found for saving.')
    
    def plot_sample(self,n=36,nrows=6,ncols=6,plot_kwargs={}):
        '''Plot samples drawn from prior for generative model.'''
        x = self.sample(n=n)
        flat_x = utils.flatten_image_batch(x, nrows=nrows, ncols=ncols)
        ax = plt.imshow(flat_x, **plot_kwargs)
        return ax


class GAN(GenerativeModel):

    '''Class for training and sampling from a generative adversarial network.
    This implementation uses Wasserstein loss with gradient penalty.'''
    def __init__(self, spec):
        super(GAN, self).__init__()
        self.spec = spec
        self.latent_dim = spec['latent_dim']
        self.image_dims = spec['image_dims']

        # Create generator and fix the input size
        with open(SPECS_DIR + spec['generative_net']) as gen_src:
            gen_spec = gen_src.read()
        gen_spec = utils.fix_batch_shape(gen_spec, [None,self.latent_dim])
        print(gen_spec)
        self.generative_net = tf.keras.models.model_from_json(gen_spec)

        # load in the discriminator
        with open(SPECS_DIR + spec['inference_net']) as inf_src:
            inf_spec = inf_src.read()
        inf_spec = utils.fix_batch_shape(inf_spec, [None] + list(self.image_dims))
        self.inference_net = tf.keras.models.model_from_json(inf_spec)

        self.d_loss_fn, self.g_loss_fn = utils.get_wgan_losses_fn()

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'], beta_1=0.5)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=spec['learning_rate'], beta_1=0.5)

    @tf.function
    def train_generator(self):

        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.spec['batch_size'], self.latent_dim ))
            x_fake = self.generative_net(z, training=True)
            x_fake_d_logit = self.inference_net(x_fake, training=True)
            G_loss = self.g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, self.generative_net.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.generative_net.trainable_variables))

        return {'g_loss': G_loss}

    @tf.function
    def train_discriminator(self,x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.spec['batch_size'], self.latent_dim))
            x_fake = self.generative_net(z, training=True)

            x_real_d_logit = self.inference_net(x_real, training=True)
            x_fake_d_logit = self.inference_net(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss =  self.d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = utils.gradient_penalty(partial(self.inference_net,
                                        training=True), x_real, x_fake)

            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.spec['gradient_penalty']

        D_grad = t.gradient(D_loss, self.inference_net.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.inference_net.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    def train(self,dataset,loss_update=100):
        t = trange(self.spec['epochs'],desc='Loss')
        for e in t:

            for j,x_real in enumerate(dataset):
                D_loss_dict = self.train_discriminator(x_real)

                if self.D_optimizer.iterations.numpy() % self.spec['gen_train_steps']== 0:
                    G_loss_dict = self.train_generator()

                if j % loss_update == 0:
                    disc_loss = D_loss_dict['d_loss']
                    gp_loss = D_loss_dict['gp']
                    gen_loss = G_loss_dict['g_loss']
                    loss_str = f'Loss - Discriminator: {disc_loss}, Generator: {gen_loss}, Gradient Penalty: {gp_loss}'
                    t.set_description(loss_str)

    def sample(self,z=None,n=100):
        if z is None:
            z = tf.random.normal(shape=(n, self.latent_dim))
        x = self.decode(z,apply_sigmoid=True)
        return x.numpy()

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    


class VAE(GenerativeModel):
    '''Class for training and sampling from a variational autoencoder'''
    def __init__(self, spec):
        super(VAE, self).__init__()
        self.spec = spec
        self.latent_dim = spec['latent_dim']
        self.image_dims = spec['image_dims']

        # Resize the last layer of the inference net
        # to have size 2 * latent_dim for both mean and log variance
        # of latent codes
        inf_spec = utils.read_and_resize(SPECS_DIR + spec['inference_net'],
                                   self.latent_dim*2, -1)
        inf_spec = utils.fix_batch_shape(inf_spec, [None] + list(self.image_dims))

        self.inference_net = tf.keras.models.model_from_json(inf_spec)

        with open(SPECS_DIR + spec['generative_net'],'r') as src:
            gen_spec = json.loads(src.read())

        # Fix number of channels from last layer of generator
        gen_spec = utils.resize_layer(gen_spec, spec['channels'],-1)
        gen_spec = utils.fix_batch_shape(gen_spec, [None,self.latent_dim])

        #gen_spec_str = json.dumps(gen_spec)
        self.generative_net = tf.keras.models.model_from_json(gen_spec)

    def sample(self, z=None, n=100):
        if z is None:
            z = tf.random.normal(shape=(n, self.latent_dim))
            x = self.decode(z, apply_sigmoid=True)
        return x.numpy()

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def train(self,dataset,loss_update=100):
        # TODO: remove this hack for using if-else cases to select
        # the optimizer
        settings = self.spec['opt_kwargs']
        if self.spec['optimizer'] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(**settings)
        else:
            raise NotImplementedError('Other optimizers are not \
                                      yet supported.')
        t = trange(self.spec['epochs'],desc='Loss')
        for i in t:
            for j,minibatch in enumerate(dataset):
                loss = compute_apply_gradients(self, minibatch,
                                        self.optimizer, vae_cross_ent_loss)
                if j % loss_update == 0:
                    t.set_description('Loss=%g' % loss)

    

@tf.function
def vae_cross_ent_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = utils.log_normal_pdf(z, 0., 0.)
    logqz_x = utils.log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer,loss_fn):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss