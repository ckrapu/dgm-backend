import numpy      as np
import tensorflow as tf
import json
import fire

from samplers import OptimizeIndependent

COMPLETION_DIR = '../data/samples/'
valid_samplers = ['independent']

def inpaint(model_path,spec_path, data_path):

    with open(spec_path,'r') as inp_spec_src:
        spec_str = inp_spec_src.read()
    spec = json.loads(spec_str)

    generative_net = tf.keras.models.load_model(model_path)

    method = spec['inpaint_method'].lower()

    if '.npy' in data_path:
        data = np.load(data_path, allow_pickle=True)
    else:
        raise ValueError('Please provide the partial images as a .npy file.')

    if method == 'independent':
        sampler = OptimizeIndependent(data, generative_net, spec)
    elif method not in valid_samplers:
        raise ValueError('Provided sampling method not \
        recognized. Please provide one of {0}.'.format(valid_samplers))

    samples = sampler.draw(spec)

    if isinstance(samples, tf.Variable):
        samples=samples.numpy()

    np.save(COMPLETION_DIR+spec['save_name'], samples)


if __name__ == '__main__':
  fire.Fire(inpaint)
