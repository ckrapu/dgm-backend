import imageio
import os

import matplotlib.pyplot as plt
import numpy as np

from tqdm import trange
from fire import Fire

from utils import flatten_image_batch, replace_batch_with_masked

def samples2gif(samples,output_path, nrows, ncols,
                directory='./',fps=5, figsize=None,masked_picture_path=None):
    '''Makes a gif out of the images listed at the indicated
    filepath.'''
    if isinstance(samples,str):
        samples = np.load(samples)
    n_samples = samples.shape[0]

    # Overlay the sampled images with the partially observed pixels
    # if desired
    if masked_picture_path is not None:
        mask_single = np.load(masked_picture_path, allow_pickle=True)

    images = []
    for i in trange(n_samples):
        x = samples[i]

        if masked_picture_path is not None:
            x = replace_batch_with_masked(mask_single,x)

        flat_x = flatten_image_batch(x, nrows, ncols)
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


def joint_gif(sample_paths,fixed_paths,fps=5):

    # Load all the gifs

    # Load fixed images

    # Get slices for common length

    # Loop and make plots
    pass


if __name__ =='__main__':
    Fire()
