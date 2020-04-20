import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

def frechet_inception_dist(classifier, synthetic, real, cut_index=-2):
    '''
    Calculates the Frechet classifier distance for a set of
    synthetic images by comparing their activations in the
    final layer of a classifier to the activations from a set
    of real images.
    '''

    assert synthetic.shape == real.shape

    # Take in pretrained classifier and
    # lop off the last layer so that it
    # outputs penultimate activations
    model_trunc = Model(classifier.inputs,
                        classifier.outputs[cut_index])

    # Fit MVN to network activations from each
    # set of images
    mu_real, sigma_real = mvn_from_trunc(model, real)
    mu_synth, sigma_synth = mvn_from_trunc(model, synthetic)

    return activation_dist(mu_real, sigma_real, mu_synth, sigma_synth)

def mvn_from_trunc(model,batch):
    '''
    Fits multivariate Gaussian to outputs of Keras model.
    '''

    activations = model(batch).numpy()
    mu = np.mean(activations,axis=0)
    sigma = np.cov(activations.T)
    return mu, sigma


def activation_dist(mu_real, sigma_real, mu_synth, sigma_synth):
    '''
    Calculates Frechet distance between multivariate normal
    distributions with known mean and covariance.
    '''

    # L2 norm of difference between mu vectors
    mu_dif_l2 = np.sum((mu_real - mu_synth)**2)

    # Square root of product of covariance matrices
    sigma_geo_mean = np.linalg.chokesky(sigma_synth @ sigma_real)
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
    '''

    y_cond = activation(classifier(images)).numpy()

    # Add very small probability to prevent 0.0 prob. values
    # Renormalize after adding probability
    y_cond = y_cond + small_prob
    y_cond = y_cond / y_cond.sum(axis=1,keepdims=True)

    y_marginal = np.mean(y_cond, axis=0, keepdims=True)
    kl_per_sample = kl_divergence(y_cond, y_marginal)
    return np.exp(kl_per_sample.mean())

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
