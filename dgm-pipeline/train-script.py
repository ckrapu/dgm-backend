import importlib
import logging
import models
import init
import tensorflow as tf

from time import time
from utils import toggle_training_layers


model_dir = '../data/saved_models/'

# Designate location of log file to list training results
# and set logging configurations
log_file = '../data/log.txt'
logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename=log_file,
                    level=logging.INFO)

# Indicate whether to save and overwrite existing
# model files
SAVE_MODELS = True

# Check to make sure that the GPU is being used
gpu_list = tf.config.list_physical_devices('GPU')
assert len(gpu_list) > 0

# Specify the datasets to be used
data_paths = [
    '../data/datasets/mnist_train.npy',
    '../data/datasets/cifar10_train.npy',
    '../data/datasets/dem32_train.npy'
]

# Indicate locations of files with model settings
'''spec_paths = [
   '../model_specs/vae_basic_mnist.json',
    '../model_specs/vae_basic_cifar10.json',
    '../model_specs/vae_basic_dem32.json'    
]'''

spec_paths = [
    '../model_specs/vae_resnet_mnist.json',
    '../model_specs/vae_resnet_cifar10.json',
    '../model_specs/vae_resnet_dem32.json'    
]

# Loop over pairs of datasets and model specifications
run_iter = zip(data_paths, spec_paths)

epochs = 10

for d, s in run_iter:

    importlib.reload(init)
    importlib.reload(models)

    start = time()
    model = init.init_from_spec(d, s, epochs=epochs)
    model.train()
    model.save(model_dir + model.spec['name'])
    toggle_training_layers(model)
    end = time()

    # Creating information for logging message
    total_time = int(end - start)
    final_loss = model.loss_history[-1]
    sn = s.split('/')[-1]
    dn = d.split('/')[-1]

    # Write message to log file
    message = f'\nTraining for model {sn} on {dn} completed after {total_time} seconds with final loss of {final_loss}.'
    logging.info(message)
