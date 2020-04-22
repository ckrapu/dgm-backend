'''
Functions for automatically creating Keras models adapted to different datasets following
general templates.
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, Flatten, LayerNormalization
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, Activation, LayerNormalization, Lambda , MaxPool2D, Dropout
from numpy import product

def identity_map():
    return lambda x: x

def identity_sequential():
    return Lambda(lambda x: x)

def residual_block(inputs, resample, filters=128, kernel_size=3,
                   normalizer=identity_map,activation='relu'):

    if resample == 'down':
        conv1 = Conv2D(filters, kernel_size=kernel_size,strides=2, padding='same')
        conv2 = Conv2D(filters, kernel_size=kernel_size,strides=1, padding='same')
        shortcut = AveragePooling2D(padding='same')

    elif resample == 'up':
        conv1 = Conv2DTranspose(filters, kernel_size=kernel_size,strides=2, padding='same')
        conv2 = Conv2D(filters, kernel_size=kernel_size,strides=1, padding='same')
        shortcut = UpSampling2D()

    elif resample == None:
        conv1 = Conv2D(filters, kernel_size=kernel_size,strides=1, padding='same')
        conv2 = Conv2D(filters, kernel_size=kernel_size,strides=1, padding='same')
        shortcut = lambda x: x

    x = conv1(inputs)
    x = Activation(activation)(x)

    x = conv2(x)

    # 1x1 convolution is used to make sure that the number of channels
    # in the identity term matches that of the transformed term.
    x = x + Conv2D(filters,kernel_size=1,padding='same')(shortcut(inputs))
    x = normalizer()(x)
    x = Activation(activation)(x)
    return x

def resnet_decoder(latent_dim, output_shape, filters=128, n_dense=2, dense_size=128,
                   n_upsample=2, blocks_per_resample=2, use_batchnorm=False,
                   use_layernorm=False,activation='relu',
                   final_activation=None):
    '''Decoder or generator using residual blocks.'''

    if use_batchnorm:
        normalizer = BatchNormalization
    elif use_layernorm:
        normalizer = LayerNormalization
    else:
        normalizer = identity_map

    h, w, c = output_shape
    h_start, w_start = int(h / (2**n_upsample)), int(w / (2**n_upsample))
    inputs = tf.keras.Input(shape=latent_dim)
    x = Lambda(lambda x: x)(inputs)

    for i in range(n_dense):
        x = Dense(dense_size)(x)
        x = normalizer()(x)
        x = Activation(activation)(x)

    x = Dense(h_start*w_start*filters,activation=activation)(x)
    x = Reshape([h_start,w_start,filters])(x)

    for i in range(n_upsample):
        filters = int(filters/2)
        x = residual_block(x, resample='up', filters=filters,
                          normalizer=normalizer,
                          activation=activation)

        for j in range(blocks_per_resample):
            x = residual_block(x, resample=None, filters=filters,
                          normalizer=normalizer,
                              activation=activation)

    x = Conv2D(kernel_size=1,filters=c,padding='same',
                      activation=final_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def resnet_encoder(output_shape, input_shape, filters=32, n_dense=2, dense_size=128,
                   n_downsample=2, blocks_per_resample=2,
                   use_batchnorm=False, use_layernorm=False,activation='relu',
                   final_activation=None):

    if use_batchnorm:
        normalizer = BatchNormalization
    elif use_layernorm:
        normalizer = LayerNormalization
    else:
        normalizer = identity_map

    inputs = tf.keras.Input(shape=input_shape)
    x = Lambda(lambda x: x)(inputs)
    for i in range(n_downsample):
        filters = int(filters*2)
        x = residual_block(x, resample='down', filters=filters,
                          normalizer=normalizer,
                          activation=activation)
        for j in range(blocks_per_resample):
            x = residual_block(x, resample=None, filters=filters,
                              normalizer=normalizer,
                              activation=activation)
    x = Flatten()(x)
    for i in range(n_dense):
        x = Dense(dense_size)(x)
        x = normalizer()(x)
        x = Activation(activation)(x)
    x = Dense(output_shape,activation=final_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def test_encoder(output_dim,image_shape,final_activation=None):
    '''Simple dense inference network used for debugging.'''
    model = tf.keras.models.Sequential()
    model.add(Input(image_shape))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(output_dim,activation=final_activation))
    return model

def test_decoder(input_dim,image_shape,final_activation=None):
    '''Simple dense generative network used for debugging.'''
    model = tf.keras.models.Sequential()
    model.add(Dense(50, input_dim=input_dim,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(int(product(image_shape)),activation=final_activation))
    model.add(Reshape(image_shape))
    return model

def conv_decoder(input_dim,image_shape,n_dense_layers=2, n_conv_per_upsample=2,
                 dense_units=128, n_upsample=2, n_conv_units_initial=256, activation='relu',
                 filter_size=3,final_activation=None,use_batchnorm=True,
                 use_layernorm=False):

    '''Automatically generates a Keras model for a simple convolutional decoder.'''

    # Determine what the dimensions of the first convolutional
    # layer need to be to have the correct image output using
    # stride-2 upsampling
    height, width, channels = image_shape
    height_initial, width_initial = int(height/2**n_upsample), int(width/2**n_upsample)

    use_bias = ~use_batchnorm

    if use_batchnorm:
        normalizer = BatchNormalization
    elif use_layernorm:
        normalizer = LayerNormalization
    else:
        normalizer = identity_sequential

    model = tf.keras.models.Sequential()

    reshape_size = n_conv_units_initial * height_initial * width_initial
    model.add(Input(input_dim))
    model.add(Dense(dense_units, input_dim=input_dim,activation=activation,use_bias=use_bias))
    model.add(normalizer())

    # Add dense portion of the network
    for i in range(n_dense_layers-1):
        model.add(Dense(dense_units,activation=activation,use_bias=use_bias))
        model.add(normalizer())

    model.add(Dense(reshape_size,activation=activation,use_bias=use_bias))
    model.add(Reshape((height_initial, width_initial,n_conv_units_initial ),input_shape=(reshape_size,)))
    model.add(normalizer())

    n_conv_units = n_conv_units_initial

    # Use alternating transpose and convolution layers to eliminate
    # checkerboard artifacts
    for i in range(n_upsample-1):
        n_conv_units = int(n_conv_units / 2)
        model.add(Conv2DTranspose(n_conv_units, filter_size, strides=2, padding='same',
                                  activation=activation,use_bias=use_bias))
        model.add(Conv2D(n_conv_units, filter_size, padding='same',activation=activation,use_bias=use_bias))
        model.add(normalizer())

        for j in range(n_conv_per_upsample):
            model.add(Conv2D(n_conv_units, filter_size, padding='same',activation=activation,use_bias=use_bias))
            model.add(normalizer())


    n_conv_units = int(n_conv_units / 2)

    model.add(Conv2DTranspose(n_conv_units, filter_size, strides=2,
                              padding='same',activation=activation,use_bias=use_bias))

    for j in range(n_conv_per_upsample):
        model.add(normalizer())
        model.add(Conv2D(n_conv_units, filter_size, padding='same',activation=activation,use_bias=use_bias))


    model.add(Conv2D(channels, 1, padding='same',activation=final_activation))
    return model

def conv_encoder(output_dim,image_shape,n_downsample=2,n_conv_per_downsample=2,
                 n_dense_layers=2,dense_units=128, n_conv_units_initial=64, activation='relu',
                 filter_size=3,use_batchnorm=False,use_layernorm=False):

    '''Counterpart encoder model for conv_decoder.'''
    model = tf.keras.models.Sequential()
    model.add(Input(image_shape))
    use_bias = ~use_batchnorm

    if use_batchnorm:
        normalizer = BatchNormalization
    elif use_layernorm:
        normalizer = LayerNormalization
    else:
        normalizer = identity_sequential

    model.add(Conv2D(n_conv_units_initial,filter_size,strides=1,
                     padding='same',activation=activation,use_bias=use_bias))
    model.add(normalizer())


    n_conv_units = 2*n_conv_units_initial

    for i in range(n_downsample):
        model.add(Conv2D(n_conv_units,filter_size,strides=2,
                         padding='same',activation=activation,use_bias=use_bias))
        model.add(normalizer())

        for j in range(n_conv_per_downsample):
            model.add(Conv2D(n_conv_units,filter_size,strides=1,
                             padding='same',activation=activation,use_bias=use_bias))
            model.add(normalizer())


        n_conv_units = n_conv_units*2

    model.add(Flatten())

    for i in range(n_dense_layers):
        model.add(Dense(dense_units,activation=activation,use_bias=use_bias))
        model.add(normalizer())

    model.add(Dense(output_dim))

    return model

def conv_classifier(input_shape,output_shape,activation='relu',
                    normalizer=BatchNormalization,n_conv_initial=32,
                    n_dense=64,dropout_prob=0.3,filter_size=5,
                    normalizer_kwarg={'momentum':0.9},n_pools=2,
                    n_conv_layer_per_pool=1,final_activation=None):
    '''Creates a CNN for classifying images. Structure is taken
    from https://github.com/ChunyuanLI/MNIS"T_Inception_Score and
    is described in "ALICE: Towards Understanding Adversarial
    Learning for Joint Distribution Matching" by Li et al. 2017
    '''

    model = tf.keras.models.Sequential()
    model.add(Conv2D(n_conv_initial, filter_size,input_shape=input_shape,
                     padding='same', activation=activation))
    model.add(normalizer(**normalizer_kwarg))
    n_conv_units = int(n_conv_initial*2)
    for i in range(n_pools):
        model.add(MaxPool2D())

        for j in range(n_conv_layer_per_pool):
            model.add(Conv2D(n_conv_units, filter_size, activation=activation))

        model.add(normalizer(**normalizer_kwarg))
        n_conv_units = int(n_conv_units*2)

    model.add(Flatten())

    # Explicitly add activation as separate layer
    # so we can extract pre-activation hidden unit values
    # for FID calculations
    model.add(Dense(n_dense))
    model.add(Activation(activation))
    model.add(Dropout(dropout_prob))
    model.add(normalizer(**normalizer_kwarg))
    model.add(Dense(output_shape, activation=final_activation))

    return model
