from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten
import sys
import keras

# content_model
def content_model(size_image, num_input_channels, content_channels_pick):

    # vgg_model = VGG19(size_image, num_input_channels)
    path = sys.path[0] + '/model/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    vgg_model = keras.applications.VGG19(weights=path)
    vgg_select_outputs = [vgg_model.get_layer(name=content).output for content in content_channels_pick]
    # feature_select = Concatenate(axis=-1)([Flatten()(vgg_select_output) for vgg_select_output in vgg_select_outputs])
    # return Model(inputs=vgg_model.input, outputs=feature_select)
    return Model(inputs=vgg_model.input, outputs=vgg_select_outputs)


def VGG19(size_image, num_input_channels):
    # vgg_path = sys.path[0] + '/model/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
    vgg_input = Input(shape=(size_image, size_image, num_input_channels))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(vgg_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return Model(vgg_input, x, name='vgg19')