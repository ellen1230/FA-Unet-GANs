from keras.layers import Input, Concatenate, Dense, Reshape
from keras.models import Model

def gdModel(generator, D_img, size_image, size_age_label, size_name_label, size_gender_label, num_input_channels):

    input_images = Input(shape=(size_image, size_image, num_input_channels))
    input_ages_conv = Input(shape=(1, 1, size_age_label))
    input_names_conv = Input(shape=(1, 1, size_name_label))
    input_genders_conv = Input(shape=(1, 1, size_gender_label))

    x = generator([input_images, input_ages_conv, input_names_conv, input_genders_conv])
    D_img.trainable = False
    gdoutput = D_img([x, input_ages_conv, input_names_conv, input_genders_conv])

    return Model(inputs=[input_images, input_ages_conv, input_names_conv, input_genders_conv], outputs=gdoutput)