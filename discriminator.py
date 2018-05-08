import keras as k
import numpy as np

from keras.layers import Input, Dense, BatchNormalization, Concatenate, initializers, Lambda
from keras.layers.core import Activation, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Model
import tensorflow as tf

from ops import lrelu, duplicate_conv

def discriminator_z_model(size_z, num_Dz_channels):
    kernel_initializer = initializers.random_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    inputs = Input(shape=(size_z,))

    # fully connection layer
    current = inputs
    for i in range(len(num_Dz_channels)):
        name = 'D_z_fc' + str(i)
        current = Dense(units=num_Dz_channels[i],
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name=name)(current)
        current = Lambda(lrelu, output_shape=(num_Dz_channels[i], ))(current)

    return Model(inputs=inputs, outputs=current)

# add the label of age + gender to input layer
def discriminator_img_model(
        size_image, size_kernel, size_age_label, size_name_label, size_gender_label, num_input_channels, num_Dimg_channels, num_Dimg_fc_channels, GANs):

    kernel_initializer = initializers.truncated_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    # Dimg model
    input_images = Input(shape=(size_image, size_image, num_input_channels))
    # the label of age + gender
    input_ages_conv = Input(shape=(1, 1, size_age_label)) # (1, 1, 10*tile_ratio)
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))
    input_ages_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_age_label),
                                   arguments={'times': size_image})(input_ages_conv) #(128, 128, 10*tile_ratio)
    input_names_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_name_label),
                                     arguments={'times': size_image})(input_names_conv)
    input_genders_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_gender_label),
                                       arguments={'times': size_image})(input_genders_conv)
    # concatenate
    current = Concatenate(axis=-1)([input_images, input_ages_conv_repeat, input_names_conv_repeat, input_genders_conv_repeat])
    
    num_layers = len(num_Dimg_channels)
    size_current = size_image
    # conv layers with stride 2
    for i in range(num_layers):
        # if i == 0:
        #     strides = 1
        # else:
        #     strides = 2
        #     size_image = int(size_image / 2)
        #
        # name = 'D_img_conv' + str(i)
        # current = Conv2D(
        #     filters=num_Dimg_channels[i],
        #     kernel_size=(size_kernel, size_kernel),
        #     strides=(strides, strides),
        #     padding='same',
        #     kernel_initializer=kernel_initializer,
        #     bias_initializer=bias_initializer,
        #     name=name)(current)

        name = 'D_img_conv' + str(i) + str('_') + str(1)
        current = Conv2D(
            filters=num_Dimg_channels[i],
            kernel_size=(size_kernel, size_kernel),
            strides=(1, 1),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        if i != 0:
            current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_image, size_image, int(current.shape[3])),
                             arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)
        current = Lambda(lrelu, output_shape=(size_image, size_image, int(current.shape[3])))(current)

        size_current = int(size_current / 2)
        name = 'D_img_conv' + str(i) + str('_') + str(2)
        current = Conv2D(
            filters=num_Dimg_channels[i],
            kernel_size=(size_kernel, size_kernel),
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        if i != 0:
            current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_current, size_current, int(current.shape[3])),
                             arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)
        current = Lambda(lrelu, output_shape=(size_current, size_current, int(current.shape[3])))(current)
        # if i < num_layers - 1:
        #     input_ages_conv_repeat = Lambda(duplicate_conv, output_shape=(size_current, size_current, size_age_label),
        #                                     arguments={'times': size_current})(input_ages_conv)  # (128, 128, 10*tile_ratio)
        #     input_names_conv_repeat = Lambda(duplicate_conv, output_shape=(size_current, size_current, size_name_label),
        #                                      arguments={'times': size_current})(input_names_conv)
        #     input_genders_conv_repeat = Lambda(duplicate_conv, output_shape=(size_current, size_current, size_gender_label),
        #                                        arguments={'times': size_current})(input_genders_conv)
        #     current = Concatenate(axis=-1)([current, input_ages_conv_repeat, input_names_conv_repeat, input_genders_conv_repeat])

    # Patch GAN_D
    # name = 'D_img_conv_final'
    # if GANs == 'LSGAN':
    #     current = Conv2D(
    #         filters=1,
    #         kernel_size=(size_kernel, size_kernel),
    #         strides=(1, 1),
    #         padding='same',
    #         kernel_initializer=kernel_initializer,
    #         bias_initializer=bias_initializer,
    #         name=name)(current)
    # elif GANs == 'cGAN':
    #     current = Conv2D(
    #         filters=1,
    #         kernel_size=(size_kernel, size_kernel),
    #         strides=(1, 1),
    #         padding='same',
    #         kernel_initializer=kernel_initializer,
    #         bias_initializer=bias_initializer,
    #         activation='sigmoid',
    #         name=name)(current)

    name = 'D_img_conv_final_1'
    current = Flatten()(current)
    current = Dense(units=num_Dimg_fc_channels, name=name)(current)
    current = Lambda(lrelu, output_shape=(num_Dimg_fc_channels, ))(current)
    name = 'D_img_conv_final_2'
    if GANs == 'LSGAN':
        current = Dense(units=1, name=name)(current)
    elif GANs == 'cGAN':
        current = Dense(units=1, activation='sigmoid', name=name)(current)

    # output
    return Model(inputs=[input_images, input_ages_conv, input_names_conv, input_genders_conv], outputs=current)


