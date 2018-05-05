from keras.layers import Input, Lambda, Concatenate, Flatten
from keras.models import Model
from keras.layers.merge import Subtract, subtract
import tensorflow as K
import numpy as np
import keras.losses


# re-write the loss_all_model
def loss_all_model(size_image, size_age_label, size_name_label, size_gender_label, num_input_channels, G, D, GANs):
    input_real_images = Input(shape=(size_image, size_image, num_input_channels))
    input_fake_images = Input(shape=(size_image, size_image, num_input_channels))

    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))


    # loss_D_real & loss_D_fake
    if GANs == 'cGAN':
        D.trainable = False
        output_D_real = D(inputs=[input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
        output_D_fake = D(inputs=[input_fake_images, input_ages_conv, input_names_conv, input_genders_conv])
        loss_D_real = Lambda(get_loss_sigmoid_cross_entropy, arguments={'labels': K.ones_like(output_D_real)}, output_shape=(None,))(output_D_real)
        loss_D_fake = Lambda(get_loss_sigmoid_cross_entropy, arguments={'labels': K.zeros_like(output_D_real)}, output_shape=(None,))(output_D_fake)

        # loss_GD
        loss_GD = Lambda(get_loss_sigmoid_cross_entropy, arguments={'labels': K.ones_like(output_D_fake)}, output_shape=(None,))(output_D_fake)
    elif GANs == 'LSGAN':
        D.trainable = False
        output_D_real = D(inputs=[input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
        output_D_fake = D(inputs=[input_fake_images, input_ages_conv, input_names_conv, input_genders_conv])
        output_D_real_compare = Lambda(K.ones_like)(output_D_real)
        output_D_fake_compare = Lambda(K.zeros_like)(output_D_fake)
        loss_D_real = Lambda(get_loss_l2, output_shape=(None,),
               arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(Subtract()([output_D_real, output_D_real_compare]))
        loss_D_fake = Lambda(get_loss_l2, output_shape=(None,),
               arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(Subtract()([output_D_fake, output_D_fake_compare]))

        # loss_GD
        output_GD_compare = Lambda(K.ones_like)(output_D_fake)
        loss_GD = Lambda(get_loss_l2, output_shape=(None,),
               arguments={'size_image': size_image, 'num_input_channels': num_input_channels})(Subtract()([output_D_fake, output_GD_compare]))


    # loss_image
    output_image = input_real_images
    output_re_image = G([input_real_images, input_ages_conv, input_names_conv, input_genders_conv])
    output_image_value = Subtract()([output_image, output_re_image])
    loss_image = Lambda(get_loss_l1, output_shape=(None,))(output_image_value)

    return Model(
        inputs=[input_real_images, input_fake_images, input_ages_conv, input_names_conv, input_genders_conv],
        outputs=[loss_D_fake, loss_D_real, loss_GD, loss_image])


def loss_seperate_model(size_batch, size_image, size_age_label, size_name_label, size_gender_label, num_input_channels, num_content_channels_pick,
                        G, D, content, GANs):
    input_compare_images = Input(shape=(size_image, size_image, num_input_channels))
    input_fake_images = Input(shape=(size_image, size_image, num_input_channels))

    inpute_compare_age_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))

    # loss_GD
    if GANs == 'cGAN':
        output_GD = D([input_fake_images, inpute_compare_age_conv, input_names_conv, input_genders_conv])
        loss_GD = Lambda(get_loss_sigmoid_cross_entropy, arguments={'labels': K.ones_like(output_GD)}, output_shape=(None,))(output_GD)
    elif GANs == 'LSGAN':
        output_GD = D([input_fake_images, inpute_compare_age_conv, input_names_conv, input_genders_conv])
        output_GD_compare = Lambda(K.ones_like)(output_GD)
        loss_GD = Lambda(mse_loss_LSGAN, output_shape=(None,),
               arguments={'size_batch': size_batch})(Subtract()([output_GD, output_GD_compare]))


    # loss_image
    output_image_value = Subtract()([input_compare_images, input_fake_images])
    loss_image = Lambda(get_loss_l1, output_shape=(None,), arguments={'size_image': 1, 'num_input_channels': 1})(output_image_value)

    # loss_content
    real_features = content(input_compare_images)
    fake_features = content(input_fake_images)
    loss_content = Lambda(get_loss_content, output_shape=(None,),
                          arguments={'fake_features': fake_features, 'num_content_channels_pick': num_content_channels_pick})(real_features)

    return Model(
        inputs=[input_compare_images, input_fake_images, inpute_compare_age_conv, input_names_conv, input_genders_conv],
        outputs=[loss_GD, loss_content, loss_image])




def get_loss_l2(t, size_image, num_input_channels):
    return K.sqrt(K.reduce_sum(K.square(t)))/size_image/size_image/num_input_channels

def get_loss_l1(t, size_image, num_input_channels):
    return K.reduce_mean(K.abs(t))/size_image/size_image/num_input_channels

def get_loss_sigmoid_cross_entropy(x, labels):
    return K.reduce_mean(K.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=labels))

def get_loss_content(real_features, fake_features, num_content_channels_pick):
    loss = 0
    for i in range(len(num_content_channels_pick)):
        (height, width, channels) = num_content_channels_pick[i]
        loss = loss + K.sqrt(K.reduce_sum(K.square(real_features[i] - fake_features[i])))/height/width
        # loss = loss + K.sqrt(K.reduce_sum(K.square(real_features[i] - fake_features[i])))
    return loss


def mse_loss_LSGAN(t, size_batch):
    loss_val = K.reduce_mean(K.sqrt(2 * K.nn.l2_loss(t)) / size_batch)
    return loss_val


