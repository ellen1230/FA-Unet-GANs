from keras.layers import Input, Dense, BatchNormalization, initializers, Concatenate, Lambda
from keras.layers.core import Activation, Reshape, Flatten, Dropout

from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Model
import tensorflow as tf

def generator_model(size_z, size_age_label, size_name_label, size_gender_label, size_mini_map, size_kernel, size_gen, num_input_channels, num_gen_channels):

    kernel_initializer = initializers.random_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    #Input layer
    input_z = Input(shape=(size_z, ))
    input_age_label = Input(shape=(size_age_label, ))
    input_name_label = Input(shape=(size_name_label,))
    input_gender_label = Input(shape=(size_gender_label,))
    current = Concatenate(axis=-1)([input_z, input_age_label, input_name_label, input_gender_label])

    # fc layer
    name = 'G_fc'
    current = Dense(
        units=size_mini_map * size_mini_map * size_gen,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)(current)
    # Reshape
    current = Reshape(target_shape=(size_mini_map, size_mini_map, size_gen))(current)
    # BatchNormalization
    #current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_mini_map, size_mini_map, size_gen),
                     #arguments={'decay': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    # Activation
    current = Activation(activation='relu')(current)

    # deconv layers with stride 2
    num_layers = len(num_gen_channels)
    size_image = size_mini_map
    for i in range(num_layers - 1):
        name = 'G_deconv' + str(i)
        current = Conv2DTranspose(
            filters=num_gen_channels[i],
            kernel_size=(size_kernel, size_kernel),
            padding='same',
            strides=(2, 2),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        # size_image = size_image * 2
        # current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_image, size_image, int(current.shape[3])),
        #                  arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)
        current = Activation(activation='relu')(current)

    # final layer of generator---> activation: tanh
    name = 'G_deconv' + str(i + 1)
    current = Conv2DTranspose(
        filters=num_gen_channels[-1],
        kernel_size=(size_kernel,size_kernel),
        padding='same',
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name=name)(current)
    current = Activation('tanh')(current)

    # output
    return Model(inputs=[input_z, input_age_label, input_name_label, input_gender_label], outputs=current)


def generator_pix2pix_model(size_image, size_age_label, size_name_label, size_gender_label, size_kernel,
                            num_input_channels, num_encoder_channels, num_gen_channels):

    from ops import duplicate_conv, lrelu
    kernel_initializer = initializers.random_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    # Encoder Input layer
    input_images = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))
    input_ages_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_age_label),
                                    arguments={'times': size_image})(input_ages_conv)
    input_names_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_name_label),
                                     arguments={'times': size_image})(input_names_conv)
    input_genders_conv_repeat = Lambda(duplicate_conv, output_shape=(size_image, size_image, size_gender_label),
                                       arguments={'times': size_image})(input_genders_conv)

    current = Concatenate(axis=-1)([input_images, input_ages_conv_repeat, input_names_conv_repeat, input_genders_conv_repeat])

    # E_conv layer + lrelu + Batch Normalization
    res_list = []
    for i in range(len(num_encoder_channels)):
        name = 'E_conv' + str(i)
        kernel_size_change = max(size_kernel - i, 2)
        current = Conv2D(
            filters=num_encoder_channels[i],
            kernel_size=(kernel_size_change, kernel_size_change),
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        size_image = int(size_image / 2)
        current = Lambda(lrelu, output_shape=(size_image, size_image, int(current.shape[3])))(current)
        current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_image, size_image, int(current.shape[3])),
                         arguments={'decay': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
        res_list.append(current)

    # G_deconv layer + Batch Normalization + relu/tanh
    size_current = current.shape[1].value
    for (i, dropout) in enumerate(num_gen_channels):

        # Residual Block ---------------> every E_conv layer has 3 mini layers(E_conv + lambda:lrelu + lambda:BN)
        if i > 0:
            # res_output = res_list[-1 - i](inputs=[input_images, input_ages_conv, input_names_conv, input_genders_conv])
            # current = Concatenate(axis=-1)([current, res_output])
            current = Concatenate(axis=-1)([current, res_list[-1-i]])

        name = 'G_deconv' + str(i)
        kernel_size_change = max(size_kernel - (len(num_gen_channels)-1- i), 2)
        current = Conv2DTranspose(
            filters=num_gen_channels[i],
            kernel_size=(kernel_size_change, kernel_size_change),
            padding='same',
            strides=(2, 2),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name)(current)
        size_current = size_current * 2
        current = Lambda(tf.contrib.layers.batch_norm, output_shape=(size_current, size_current, int(current.shape[3])),
                         arguments={'decay':0.9, 'epsilon': 1e-5, 'scale':True})(current)
        if dropout > 0.0:
            current = Dropout(rate=dropout)

        if i == len(num_gen_channels)-1:
            current = Activation(activation='tanh')(current)
        else:
            current = Activation(activation='relu')(current)

    # output
    return Model(inputs=[input_images, input_ages_conv, input_names_conv, input_genders_conv], outputs=current)
