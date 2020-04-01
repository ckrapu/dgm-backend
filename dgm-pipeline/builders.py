import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, Flatten

'''Code for automatically creating Keras models adapted to different datasets following
general templates.'''

def conv_decoder(input_dim,image_shape,n_dense_layers=2,
                 dense_units=128, n_upsample=2, n_conv_units_initial=128, activation='relu',
                 filter_size=5,final_activation='sigmoid',use_batchnorm=False):
    
    '''Automatically generates a Keras model for a simple convolutional decoder.'''
    
    # Determine what the dimensions of the first convolutional
    # layer need to be to have the correct image output using
    # stride-2 upsampling
    height, width, channels = image_shape
    height_initial, width_initial = int(height/2**n_upsample), int(width/2**n_upsample)
    
    use_bias = ~use_batchnorm
    
    model = tf.keras.models.Sequential()
    
    reshape_size = n_conv_units_initial * height_initial * width_initial
    model.add(Dense(dense_units, input_dim=input_dim,activation=activation,use_bias=use_bias))
    if use_batchnorm: model.add(BatchNormalization())
    
    # Add dense portion of the network
    for i in range(n_dense_layers-1):
        model.add(Dense(dense_units,activation=activation,use_bias=use_bias))
        if use_batchnorm: model.add(BatchNormalization())
           
    model.add(Dense(reshape_size,activation=activation,use_bias=use_bias))              
    model.add(Reshape((height_initial, width_initial,n_conv_units_initial ),input_shape=(reshape_size,)))
    if use_batchnorm: model.add(BatchNormalization())

    n_conv_units = n_conv_units_initial
    
    # Use alternating transpose and convolution layers to eliminate 
    # checkerboard artifacts
    for i in range(n_upsample-1):
        model.add(Conv2DTranspose(n_conv_units, filter_size, strides=2, padding='same',activation=activation,use_bias=use_bias))
        if use_batchnorm: model.add(BatchNormalization())

        model.add(Conv2D(n_conv_units, filter_size, padding='same',activation=activation,use_bias=use_bias))
        if use_batchnorm: model.add(BatchNormalization())

        n_conv_units = int(n_conv_units / 2)
        
    model.add(Conv2DTranspose(n_conv_units, filter_size, strides=2, padding='same',activation=activation,use_bias=use_bias))
    if use_batchnorm: model.add(BatchNormalization())

    model.add(Conv2D(channels, 3, padding='same',activation=final_activation))
    return model

def conv_encoder(output_dim,image_shape,n_downsample=2,n_dense_layers=2,dense_units=128, n_conv_units_initial=64, activation='relu',
                 filter_size=5,use_batchnorm=False):

    '''Counterpart encoder model for conv_decoder.'''
    model = tf.keras.models.Sequential()
    model.add(Input(image_shape))
    use_bias = ~use_batchnorm
    
    n_conv_units = n_conv_units_initial
    for i in range(n_downsample):
        model.add(Conv2D(n_conv_units,filter_size,strides=2,padding='same',activation=activation,use_bias=use_bias))
        if use_batchnorm: model.add(BatchNormalization())
        n_conv_units = n_conv_units*2
    
    model.add(Flatten())
    
    for i in range(n_dense_layers):
        model.add(Dense(dense_units,activation=activation,use_bias=use_bias))
        if use_batchnorm: model.add(BatchNormalization())
            
    model.add(Dense(output_dim))
    
    return model
