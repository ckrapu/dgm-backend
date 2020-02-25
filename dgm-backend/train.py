import fire
import json
import numpy as np

from models import VAE, GAN
from utils  import describe_spec, get_diagnostics, prep_data


SAVED_MODELS_DIR = '../data/saved_models/'
DTYPE            = 'float32'

model_mapping= {'vae':VAE,'gan':GAN}

def init_model(spec):

    valid_models = model_mapping.keys()
    model_name = spec['model_type']

    if model_name in valid_models:
        initializer = model_mapping[model_name]
        return initializer(spec)

    else:
        raise NotImplementedError('Model type {0} is not supported.\
                                  Please try one of {1}'.format(model_name,
                                  valid_models))

def train(data_path, model_spec_path):

  # load data
  if 'npy' in data_path:
    data = np.load(data_path).astype(DTYPE)
  else:
    raise NotImplementedError('Only datasets with the .npy extension are \
                              supported.')

  with open(model_spec_path,'r') as spec_src:
      spec_as_str = spec_src.read().strip()

  spec = json.loads(spec_as_str)
  spec.update({'image_dims':data.shape[1:],'channels':data.shape[-1]})
  describe_spec(spec)

  dataset = prep_data(data,spec)

  model = init_model(spec)

  model.train(dataset)
  model.save(SAVED_MODELS_DIR + spec['name'])

  get_diagnostics(model)

  return

if __name__ == '__main__':
  fire.Fire(train)