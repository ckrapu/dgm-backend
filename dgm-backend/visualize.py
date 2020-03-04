import imageio
import os

import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange
from fire import Fire

from utils import flatten_image_batch

def samples2gif(samples_path,output_path, nrows, ncols,
                directory='./',fps=5, figsize=None):
    '''Makes a gif out of the images listed at the indicated
    filepath.'''
    samples = np.load(samples_path)
    n_samples = samples.shape[0]

    images = []
    for i in trange(n_samples):
        x = samples[i]
        flat_x = flatten_image_batch(x.squeeze(), nrows, ncols)
        filepath = directory + f'gif_frame_{i}.png'

        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(flat_x)
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        images.append(imageio.imread(filepath))
        os.remove(filepath)
    imageio.mimsave(output_path,images,fps=fps)

if __name__ =='__main__':
    Fire()
