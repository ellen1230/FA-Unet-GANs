from keras.layers import Input, Lambda, Concatenate
from keras.models import Model
from keras.layers.merge import Subtract, subtract
import tensorflow as K
import numpy as np
import keras.losses


# re-write the loss_all_model
def loss_all_model(size_image, size_age_label, size_name_label, size_gender_label, num_input_channels, G, D):
    input_real_images = Input(shape=(size_image, size_image, num_input_channels))
    input_fake_images = Input(shape=(size_image, size_image, num_input_channels))

    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))


    # loss_D_real & loss_D_fake
    D.trainable = False
    output_D_real = D(inputs=[input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_D_fake = D(inputs=[input_fake_images, input_ages_conv, input_names_conv, input_genders_conv])
    loss_D_real = Lambda(get_loss_D, arguments={'labels': K.ones_like(output_D_real)}, output_shape=(None,))(output_D_real)
    loss_D_fake = Lambda(get_loss_D, arguments={'labels': K.zeros_like(output_D_real)}, output_shape=(None,))(output_D_fake)

    # loss_GD
    loss_GD = Lambda(get_loss_D, arguments={'labels': K.ones_like(output_D_fake)}, output_shape=(None,))(output_D_fake)

    # loss_image
    output_image = input_real_images
    output_re_image = G([input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_image_value = Subtract()([output_image, output_re_image])
    loss_image = Lambda(get_loss_l1, output_shape=(None,),
                        arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(output_image_value)

    return Model(
        inputs=[input_real_images, input_fake_images, input_ages_conv, input_names_conv, input_genders_conv],
        outputs=[loss_D_fake, loss_D_real, loss_GD, loss_image])


def loss_seperate_model(size_image, size_age_label, size_name_label, size_gender_label, num_input_channels, G, D):
    input_real_images = Input(shape=(size_image, size_image, num_input_channels))
    input_fake_images = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))

    # loss_GD
    output_GD = D([input_fake_images, input_ages_conv, input_names_conv, input_genders_conv])
    loss_GD = Lambda(get_loss_D, arguments={'labels': K.ones_like(output_GD)}, output_shape=(None,))(output_GD)

    # loss_image
    output_image = input_real_images
    output_re_image = G([input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_image_value = Subtract()([output_image, output_re_image])
    loss_image = Lambda(get_loss_l1, output_shape=(None,), arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(
        output_image_value)

    return Model(
        inputs=[input_real_images, input_fake_images, input_ages_conv, input_names_conv, input_genders_conv],
        outputs=[loss_GD, loss_image])




def get_loss_l2(t, size_image, num_input_channels):
    if int(num_input_channels) == 1:
        return K.sqrt(K.reduce_sum(K.square(t)))/size_image
    else:
        return K.sqrt(K.reduce_sum(K.square(t)))/size_image/size_image/num_input_channels

def get_loss_l1(t, size_image, num_input_channels):
    if int(num_input_channels) == 1:
        return K.reduce_mean(K.abs(t))
    else:
        return K.reduce_mean(K.abs(t))

def get_loss_D_real(x):
    K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=K.ones_like(x)))
    import tensorflow


def get_loss_D_fake(x):
    K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=K.zeros_like(x)))

def get_loss_binary(x, binary):
    from keras.losses import binary_crossentropy
    return binary_crossentropy(x, binary)

def get_loss_D(x, labels):
    return K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=labels))

