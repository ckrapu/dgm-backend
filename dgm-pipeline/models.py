import json
import utils
import builders as bld
import quality  as q

import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from tqdm      import trange, tqdm
from os        import remove
from os.path   import exists

builder_mapping = {'conv_decoder':bld.conv_decoder,
                   'conv_encoder':bld.conv_encoder,
                   'test_decoder':bld.test_decoder,
                   'test_encoder':bld.test_encoder,
                   'resnet_decoder':bld.resnet_decoder,
                   'resnet_encoder':bld.resnet_encoder}

SPECS_DIR = '../model_specs/network_specs/'
VIZ_DIR   = '../data/visualizations/'  

class GenerativeModel(tf.keras.Model):
    '''Common class for deep generative models'''
    def __init__(self,spec):
        super().__init__()

        self.spec = spec
        self.latent_dim = spec['latent_dim']
        self.image_dims = spec['image_dims']

    def create_generator(self):
        '''
        Initialize generative network either via reading
        a JSON spec or using a function to initialize it.
        '''
        # Create generator and fix the input size
        if 'json' in self.spec['generative_net']:
            with open(SPECS_DIR + self.spec['generative_net']) as gen_src:
                gen_spec = gen_src.read()
            gen_spec = utils.fix_batch_shape(gen_spec, [None,self.latent_dim])
            self.generative_net = tf.keras.models.model_from_json(gen_spec)
        else:
            network_builder = builder_mapping[self.spec['generative_net']]
            kw = self.spec['generative_net_kwargs']
            self.generative_net = network_builder(self.latent_dim, self.image_dims, **kw)
    
    def create_inference(self,output_shape=1):
        '''
        Initialize inference/decoder network either via reading
        a JSON spec or using a function to initialize it.
        '''
        if 'json' in self.spec['inference_net']:
            with open(SPECS_DIR + self.spec['inference_net']) as inf_src:
                inf_spec = inf_src.read()

            inf_spec = utils.fix_batch_shape(inf_spec, [None]+ list(self.image_dims))
            self.inference_net = tf.keras.models.model_from_json(inf_spec)
        else:
            network_builder = builder_mapping[self.spec['inference_net']]
            kw = self.spec['inference_net_kwargs']

            self.inference_net = network_builder(output_shape, self.image_dims, **kw)

    def load_pretrained(self, gen_path=None, inf_path=None):
        '''
        Use pretrained Keras model for either generative or inference
        network.
        '''
        if gen_path is not None:
            self.generative_net = tf.keras.models.load_model(gen_path)
        
        if inf_path is not None:
            self.inference_net = tf.keras.models.load_model(inf_path)

    def save(self, prefix, overwrite=True):
        '''
        Save networks to disk via Keras utilities.
        '''
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
        '''
        Plot samples drawn from prior for generative model.
        '''
        x = self.sample(n=n)
        flat_x = utils.flatten_image_batch(x, nrows=nrows, ncols=ncols)
        ax = plt.imshow(flat_x, **plot_kwargs)
        return ax

    def test_batch(self):
        if hasattr(self,'dataset'):
            iterator = self.dataset.as_numpy_iterator()
            return next(iterator)

        else:
            raise ValueError('Dataset has not been set for this model yet.')

    def create_masked_logp_fn(self,masked_batch,loss_elemwise_fn,loss_kwargs={},
                              final_activation_fn=None,dtype='float32',temperature=1.):
        '''
        Applies masking to the logged posterior for a set of images
        according.
        '''
        is_masked = masked_batch.mask
        is_used   = tf.cast(1 - is_masked,dtype)
        raw_data = tf.cast(masked_batch.data, dtype)

        def logp(z):
            x = self.generative_net(z)
            if final_activation_fn is not None:
                x = final_activation_fn(x)

            loss_elemwise = loss_elemwise_fn(raw_data, x, **loss_kwargs)

            # The argument to reduce sum should have 4 dimensions
            loglik = -tf.reduce_sum(loss_elemwise * is_used, axis=[1,2,3])

            # We can use hot / cold posteriors by altering
            # the temperature value
            return temperature*loglik + self.log_prior_fn(z)

        return logp

    def inception_score(self, classifier_path,n=10000):
        '''
        Calculates the inception score for this model
        using an externally trained classifier.
        '''

        if not hasattr(self,'scores'):
            self.scores = {}

        xs = self.sample(n)
        iscore = qq.inception_score(classifier_path, xs)
        self.scores['inception_score'] = iscore
        return iscore
        

class GAN(GenerativeModel):
    '''
    Class for training and sampling from a generative adversarial network.
    This implementation uses Wasserstein loss with gradient penalty (WGAN-GP).
    '''
    def __init__(self, spec):
        super().__init__(spec)
        

        self.create_generator()
        self.create_inference(output_shape=1)

        # Assumes that batch of z variable will have shape
        # [batch_size X latent_dim]
        self.log_prior_fn = lambda z: -tf.reduce_sum(z**2,axis=-1)/2
    
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

    def train(self,loss_update=100):
        
        self.loss_history = []
        for e in range(self.spec['epochs']):
            t = tqdm(enumerate(self.dataset),desc='Loss')
            for j,x_real in t:
                D_loss_dict = self.train_discriminator(x_real)

                if self.D_optimizer.iterations.numpy() % self.spec['gen_train_steps']== 0:
                    G_loss_dict = self.train_generator()

                if j % loss_update == 0 and j > self.spec['gen_train_steps']:
                    disc_loss = D_loss_dict['d_loss']
                    gp_loss = D_loss_dict['gp']
                    gen_loss = G_loss_dict['g_loss']
                    loss_str = f'Loss - Discriminator: {disc_loss}, Generator: {gen_loss}, Gradient Penalty: {gp_loss}'
                    t.set_description(loss_str)
                    self.loss_history.append([disc_loss, gp_loss, gen_loss])

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
        super().__init__(spec)

        self.create_generator()
        self.create_inference(output_shape=self.latent_dim*2)

        # Assumes that batch of z variable will have shape
        # [batch_size X latent_dim]
        self.log_prior_fn = lambda z: -tf.reduce_sum(z**2,axis=-1)/2

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

    def set_beta_schedule(self,schedule):
        self.beta_schedule = schedule

    @staticmethod
    @tf.function
    def compute_apply_gradients(model, optimizer, x, loss_fn,beta=1.):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, x, beta=beta)
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def train(self,loss_update=100,error_trainable=True):

        # TODO: remove this hack for using if-else cases to select
        # the optimizer
        settings = self.spec['opt_kwargs']
        if self.spec['optimizer'] == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(**settings)
        else:
            raise NotImplementedError('Other optimizers are not \
                                      yet supported.')
        t = trange(self.spec['epochs'],desc='Loss')
        self.loss_history = []

        # Control representation capacity per Burgess et al. 2018
        # 'Understanding disentangling in beta-VAE'
        if 'vae_beta' in self.spec.keys():
            beta = self.spec['vae_beta']
        else:
            beta = 1.
        
        loglik_type = self.spec['likelihood']

        if loglik_type == 'bernoulli':
            loglik = cross_ent_loss

        elif loglik_type == 'normal':
            # Enables the error sd to be variable and
            # learned by the data
            if self.spec['error_trainable']:
                self.error_sd = tf.Variable(0.1)
            else:
                self.error_sd = 0.1
            loglik = partial(square_loss, sd=self.error_sd)
        else:
            raise ValueError('Likelihood argument not understood. Try one of "bernoulli" or "normal".')

        vae_loss_fn = partial(vae_loss, loglik=loglik)

        for i in t:
            for j,minibatch in enumerate(self.dataset):
                if hasattr(self,'beta_schedule'):
                    beta_current = self.beta_schedule[i]
                else:
                    beta_current = beta
                loss = self.compute_apply_gradients(self, self.optimizer, 
                                                    minibatch, vae_loss_fn, beta=beta_current)
                if j % loss_update == 0:
                    t.set_description('Loss=%g' % loss)
                self.loss_history.append(loss)


@tf.function
def square_loss(x_pred, x_true, sd=1, axis=[1,2,3,]):
    error = (x_pred-x_true)**2 / (2*sd**2)
    return -tf.reduce_sum(error,axis=axis)   

@tf.function
def cross_ent_loss(x_logit, x_label, axis=[1,2,3]):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x_label)
    loss = -tf.reduce_sum(cross_ent, axis=axis)
    return loss

#TODO: Implement the beta likelihood
@tf.function
def beta_loss(a, b, x_label, axis=[1,2,3]):

    pass

@tf.function
def vae_loss(model, x, loglik, beta=1.):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)
    logpx_z = loglik(x_pred, x)
    logpz = utils.log_normal_pdf(z, 0., 0.)
    logqz_x = utils.log_normal_pdf(z, mean, logvar)
    kld = logqz_x - logpz
    return -tf.reduce_mean(logpx_z - beta * kld)

@tf.function
def vae_cross_ent_loss(model, x, beta=1.):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    logpx_z = cross_ent_loss(x_logit, x)
    logpz = utils.log_normal_pdf(z, 0., 0.)
    logqz_x = utils.log_normal_pdf(z, mean, logvar)
    kld = logqz_x - logpz
    return -tf.reduce_mean(logpx_z - beta*kld)

@tf.function
def wrapped_cross_ent(true,pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=true)
