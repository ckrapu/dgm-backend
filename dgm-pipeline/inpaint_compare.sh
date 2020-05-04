#!/bin/bash

MNIST_DATA_PATH=../../data/mnist.npy

# Train all models on MNIST data
#python train.py $MNIST_DATA_PATH ../model_specs/gan_mnist.json 
python train.py $MNIST_DATA_PATH ../model_specs/vae_mnist.json

# Apply optimization, LD and MALA to inpainting problem

# Create GIF from samples from each set of samples