import numpy      as np
import tensorflow as tf
import json
import fire

from samplers import LangevinDynamics

COMPLETION_DIR = '../data/samples/'
valid_samplers = ['langevin']

def inpaint(model_path, spec_path, data_path,save=True,**kwargs):
    '''Uses a deep generative model to conduct inpainting for partially
    obscured images.

    Arguments
    ---------
    model_path : string
        Filepath to a saved Tensorflow Model file containing the weights for
        a generative model
    spec_path : string
        Filepath for a JSON file that stores the configurations for the
        inpainting algorithm
    data_path : string
        Filepath for a masked Numpy array saved in .npy format which
        contains the partially obscured images to be inpainted
    '''

    with open(spec_path,'r') as inp_spec_src:
        spec_str = inp_spec_src.read()
    spec = json.loads(spec_str)

    generative_net = tf.keras.models.load_model(model_path)

    method = spec['inpaint_method'].lower()

    if '.npy' in data_path:
        data = np.load(data_path, allow_pickle=True)
    else:
        raise ValueError('Please provide the partial images as a .npy file.')

    
    for key,value in kwargs.items():
        if key in spec.keys():
            spec[key] = value
    if method == 'langevin':
        sampler = LangevinDynamics(data, generative_net, spec)
    elif method not in valid_samplers:
        raise ValueError('Provided sampling method not recognized. Please provide one of {0}.'.format(valid_samplers))
    
    samples = sampler.draw(spec)

    if isinstance(samples, tf.Variable):
        samples=samples.numpy()

    if save:
        np.save(COMPLETION_DIR+spec['save_name'], samples)

    return samples

if __name__ == '__main__':
  fire.Fire(inpaint)
