import numpy as np
import tensorflow as tf

from warnings import warn
from tensorflow.keras.models import Model

 
def frechet_inception_dist(classifier, synthetic, real, cut_index=-2):
    '''
    Calculates the Frechet classifier distance for a set of
    synthetic images by comparing their activations in the
    final layer of a classifier to the activations from a set
    of real images.

    Arguments
    ---------
    classifer : str or Keras Model
        Classification model or filepath for a saved Keras model.
    synthetic : Numpy array
        Array of simulated images with shape [n, h, w, channels]
    real : Numpy array
        Array of real images with same shape as previous
    cut_index : int
        Index into Keras model to select layer. To select the
        second-to-last layer, leave as the default value of -2.

    Returns
    -------
    fid : float
        Frechet inception distance for the simulated images

    '''

    assert synthetic.shape == real.shape

    if isinstance(classifier,str):
        classifier = tf.keras.models.load_model(classifier)
        
    # Take in pretrained classifier and
    # lop off the last layer so that it
    # outputs penultimate activations
    model_trunc = Model(classifier.inputs,
                        classifier.layers[cut_index].output)

    # Fit MVN to network activations from each
    # set of images
    real = real.astype('float64')
    synthetic = synthetic.astype('float64')

    mu_real, sigma_real = mvn_from_trunc(model_trunc, real)
    mu_synth, sigma_synth = mvn_from_trunc(model_trunc, synthetic)

    try:
        fid = activation_dist(mu_real.astype('float64'), 
                            sigma_real.astype('float64'),
                             mu_synth.astype('float64'),
                              sigma_synth.astype('float64'))
        return fid

    except np.linalg.linalg.LinAlgError:
        warn('Cholesky decomposition failed. Returning covariance matrices instead.')
        return sigma_real, sigma_synth
        

def mvn_from_trunc(model,batch,eps=1e-3):
    '''
    Fits multivariate Gaussian to outputs of Keras model.

    Arguments
    ---------
    model : Keras Model
        Used to generate activations from batch of images
    batch : Numpy array
        Inputs into the model with shape [n, ...]

    Returns
    -------
    mu : 1D Numpy array
        Sample mean vector of MVN fit to activations
    sigma : 2D Numpy array
        Sample covariance of activations
    '''

    activations = model.predict(batch)
    mu = np.mean(activations,axis=0)

    # The covariance matrix is sometimes singular so 
    # a small amount is added to the diagonal.
    print(activations.shape)
    sigma = np.cov(activations.T) + np.eye(mu.shape[0])*eps
    return mu, sigma


def activation_dist(mu_real, sigma_real, mu_synth, sigma_synth):
    '''
    Calculates Frechet distance between multivariate normal
    distributions with known mean and covariance.
    '''

    # L2 norm of difference between mu vectors
    mu_dif_l2 = np.sum((mu_real - mu_synth)**2)

    # Square root of product of covariance matrices
    sigma_geo_mean = np.linalg.cholesky(sigma_synth @ sigma_real)
    sigma_add = sigma_real+sigma_synth

    # Formula from Heusel et al. 2017
    fid = mu_dif_l2 + np.trace(sigma_add - 2*sigma_geo_mean)
    return fid

def inception_score(classifier, images, activation=tf.math.softmax,
                    small_prob = 1e-10):
    '''
    Calculates the inception score for a set of images
    using a pretrained classifier. See "A Note on the Inception
    Score" by Barratt and Sharma (2018) for more details.

    Arguments
    ---------
    classifer : str or Keras Model
        Classification model or filepath for a saved Keras model.
    images : Numpy array
        Batch of images from generative model
    activation : function
        Function to be applied to samples before feeding into
        classifier
    small_prob : float
        Very small value used to adjust 0.0 probability values
        upwards to avoid computations involving log(0.)

    Returns
    -------
    inception_score : float
        Number indicating the average KL divergence between
        the marginal distribution of labels and per-datapoint
        distribution of conditional labels. Higher values are
        better.
    '''

    if isinstance(classifier,str):
        classifier = tf.keras.models.load_model(classifier)

    y_cond = activation(classifier(images)).numpy()

    # Add very small probability to prevent 0.0 prob. values
    # Renormalize after adding probability
    y_cond = y_cond + small_prob
    y_cond = y_cond / y_cond.sum(axis=1,keepdims=True)

    y_marginal = np.mean(y_cond, axis=0, keepdims=True)
    kl_per_sample = kl_divergence(y_cond, y_marginal)
    inception_score = np.exp(kl_per_sample.mean())
    return inception_score

def kl_divergence(p,q,sum_axis=1):
    '''
    Calculates vectorized Kullback-Leibler divergence
    between distributions with identical, discrete support.
    Default settings assume that first dimension ranges
    over samples, while second dimension ranges over dimensions
    of the support.
    '''

    elemwise = p*(np.log(p)-np.log(q))
    return elemwise.sum(axis=sum_axis)
