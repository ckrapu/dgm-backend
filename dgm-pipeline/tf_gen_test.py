import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import * 

latent_dim = 10

layers = [
    Input(shape=(latent_dim)),
    Dense(256, activation=ReLU),
    Dense(256, activation=ReLU),
    Dense(7*7*128, activation=ReLU),
    Reshape([None,7,7,128]),
    Conv2DTranspose(128,3,strides=(2,2),padding='same',activation=ReLU),
    Conv2DTranspose(128,3,padding='same',activation=ReLU),
    Conv2DTranspose(64,3,strides=(2,2),padding='same',activation=ReLU),
    Conv2DTranspose(64,3,padding='same',activation=ReLU),
    Conv2DTranspose(64,3,strides=(2,2),padding='same',activation=ReLU),
    Conv2DTranspose(1,3,padding='same',activation=ReLU)
        ]
generator = tf.keras.models.Sequential(layers)
print(generator.summary())
