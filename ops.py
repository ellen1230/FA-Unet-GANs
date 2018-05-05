from scipy.misc import imread, imresize, imsave
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
from keras.layers import Concatenate
from PIL import Image
import scipy.io as scio
import random
import keras
from pylab import *
import matplotlib as mpl
# mpl.use('Agg')

def load_sample(
        image_path,  # path of a image
        image_size,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
        ):
    if is_gray:
        image = imread(image_path, flatten=True).astype(np.float32)
    else:
        image = imread(image_path).astype(np.float32)
        # image = np.array(Image.open(image_path).resize((image_size, image_size)))

    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image

def load_image(filenames, size_image, size_scale, num_input_channels):

    # path_queue = tf.train.string_input_producer(image_path, shuffle=True)
    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(path_queue)
    images = []

    for filename in filenames:
        file_contents = tf.read_file(filename)
        image = tf.image.decode_jpeg(file_contents)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image * 2 - 1

        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_images(image, [size_scale, size_scale], method=tf.image.ResizeMethod.BILINEAR)
        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, size_scale - size_image + 1)), dtype=tf.int32)
        image = tf.image.crop_to_bounding_box(image, offset[0], offset[1], size_image, size_image)
        image.set_shape([size_image, size_image, num_input_channels])
        images.append(image)

    return images

def age_group_label(label):
    if 0 <= label <= 5:
        label = 0
    elif 6 <= label <= 10:
        label = 1
    elif 11 <= label <= 15:
        label = 2
    elif 16 <= label <= 20:
        label = 3
    elif 21 <= label <= 30:
        label = 4
    elif 31 <= label <= 40:
        label = 5
    elif 41 <= label <= 50:
        label = 6
    elif 51 <= label <= 60:
        label = 7
    elif 61 <= label <= 70:
        label = 8
    else:
        label = 9
    return label

# duplicate the label of age + gender to tile_ratio times
def duplicate(enable_tile_label, tile_ratio, size_age):
    # age duplicate (tile_ratio) times
    if enable_tile_label:
        size_label = int(size_age * tile_ratio)
    else:
        size_label = size_age
    # # gender duplicate (tile_ratio) times
    # if enable_tile_label:
    #     size_label = size_label + int(2 * tile_ratio)
    # else:
    #     size_label = size_label + 2

    return size_label


def concat_label(label, enable_tile_label, tile_ratio):
    if enable_tile_label:
        for i in range(int(tile_ratio)-1):
            label = np.concatenate((label, label), axis=-1)

    return label
    #
    # x_shape = x.get_shape().as_list()
    # if duplicate < 1:
    #     return x
    # # duplicate the label to enhance its effect, does it really affect the result?
    # label = tf.tile(label, [1, duplicate])
    # label_shape = label.get_shape().as_list()
    # if len(x_shape) == 2:
    #     return tf.concat([x, label], 1)
    # elif len(x_shape) == 4:
    #     label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
    #     return tf.concat([x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])], 3)


def load_weights(save_dir, EGD):
    print("\n\tLoading pre-trained model ...")
    EGD = EGD.load_weights(str(save_dir) + '/EGD.h5', by_name=True)
    return EGD

def save_weights(save_dir, GD_model, content_model, epoch, batch):
    print("\n\tsaving trained model_e", epoch, " ...")
    GD_model.save(filepath=save_dir +"/GD_" + str(epoch) + 'b' + str(batch) + ".h5", overwrite=True)
    content_model.save(filepath=save_dir + "/content_" + str(epoch) + 'b' + str(batch) + ".h5", overwrite=True)
    return "SUCCESS!"

def save_image(images, size_image, image_value_range, num_input_channels, epoch, batch, mode, image_path):
    print("\n\tsaving generated images_e", epoch, " ...")
    num_images = len(images)
    if num_images < 16:
        num_picture = int(np.sqrt(num_images))
        picture = np.zeros([size_image * num_picture, size_image * num_picture, num_input_channels])

        for i in range(num_picture):
            for j in range(num_picture):
                index = i * num_picture + j
                picture[i * size_image:(i + 1) * size_image, j * size_image:(j + 1) * size_image] = images[index]
    else:
        num_picture = 4
        picture = np.zeros([size_image * num_picture, size_image * num_picture, num_input_channels])

        for i in range(num_picture):
            for j in range(num_picture):
                index = i * num_picture + j
                picture[i * size_image:(i + 1) * size_image, j * size_image:(j + 1) * size_image] = images[index]




    # picture = ((picture + 1)/2 * 255.0).astype(np.uint8)
    picture = ((picture + 1) / 2).astype(np.float32)
    path = str(image_path)  + "/e" + str(epoch) + 'b' + str(batch) + ".png"
    imsave(path, picture)



def save_loss(save_dir, loss_Dimg, loss_D, loss_GD, loss_content, loss_image, loss_all):


    np.save(save_dir + '/loss_Dimg(all).npy', np.array(loss_D))
    np.save(save_dir+'/loss_Dimg.npy', np.array(loss_Dimg))
    np.save(save_dir + '/loss_GD.npy', np.array(loss_GD))
    np.save(save_dir + '/loss_content.npy', np.array(loss_content))
    np.save(save_dir + '/loss_image.npy', np.array(loss_image))
    np.save(save_dir+'/loss_all.npy', np.array(loss_all))


def copy_array(array, times):
    a = []
    for i in range(times):
        a.append(array.tolist())
    return np.array(a)


def duplicate_conv(x, times):
    list = []
    for i in range(times):
        list.append(x)
    x = keras.layers.Concatenate(axis=1)(list)
    list = []
    for i in range(times):
        list.append(x)
    output = keras.layers.Concatenate(axis=2)(list)
    return output

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def name_gender_label(file_name):
    names = ['Kurt_Russell', 'Olivia_Hussey', 'Robin_Williams', 'Mark_Hamill', 'Jane_Seymour', 'Lynda_Carter', 'Michael_Cera', 'Julianna_Margulies', 'Treat_Williams', 'Peter_Jackson', 'Roseanne_Barr', 'William_Katt', 'Sam_Raimi', 'Rosanna_Arquette', 'Tilda_Swinton', 'Tim_Daly', 'Dana_Delany', 'Jennifer_Tilly', 'Kevin_Pollak', 'Madonna', 'Tom_Sizemore', 'John_Goodman', 'Vicki_Lewis', 'Virginia_Madsen', 'Woody_Harrelson', 'Zeljko_Ivanek', 'Adam_Arkin', 'Alfre_Woodard', 'Jon_Bon_Jovi', 'Julia_Louis-Dreyfus', 'ulianna_Margulies', 'Julianne_Moore', 'Ray_Romano', 'Peter_Jackson', 'Matt_LeBlanc', 'Fran_Drescher', 'Hugh_Grant', 'Geena_Davis', 'Eric_Roberts', 'Kevin_Spacey', 'Jeff_Perry', 'Jennifer_Coolidge', 'Dylan_McDermott', 'Patricia_Clarkson', 'Jennifer_Tilly', 'Mykelti_Williamson', 'Michael_Cerveris', 'Melanie_Griffith', 'Kelsey_Grammer', 'Lauren_Graham', 'Mark_Wahlberg', 'Megyn_Price', 'Nathan_Fillion', 'Paget_Brewster', 'Paul_Rudd', 'Virginia_Madsen', 'Sean_Patrick_Flanery', 'Shannen_Doherty', 'Taraji_P._Henson', 'Tim_Guinee', 'Rachel_Weisz', 'Stephen_Baldwin', 'Mark_Ruffalo', 'Tina_Fey', 'Aaron_Eckhart', 'Adewale_Akinnuoye-Agbaje', 'Sharlto_Copley', 'Amanda_Peet', 'Bradley_Cooper', 'Carla_Gallo', 'Casper_Van_Dien', 'Christina_Hendricks', 'Maya_Rudolph', 'Martin_Freeman', 'Naomi_Watts', 'Denise_Richards', 'Edward_Norton', 'Elizabeth_Mitchell', 'Michael_Pitt', 'Emily_Procter', 'Lucy_Liu', 'Aaron_Johnson', 'Britt_Robertson', 'Caitlin_Stasey', 'Chris_Colfer', 'Diego_Boneta', 'Emma_Watson', 'Jennifer_Lawrence', 'Liam_Hemsworth', 'Rupert_Grint', 'Rose_McIver', 'Naya_Rivera', 'Daniel_Radcliffe', 'Amanda_Bynes', 'Brant_Daugherty', 'Dakota_Johnson', 'Hunter_Parrish', 'Emma_Stone', 'Kristen_Connolly', 'Keira_Knightley', 'Taylor_Kinney', 'Hayden_Christensen', 'Katy_Perry', 'January_Jones', 'Chace_Crawford', 'Lily_Rabe', 'Matt_Bomer', 'Masiela_Lusha', 'Matthew_Goode', 'Nick_Cannon', 'Britney_Spears', 'Aimee_Teegarden', 'Jesse_Tyler_Ferguson', 'Drea_de_Matteo', 'Corey_Haim', 'Shawn_Ashmore', 'Sarah_Drew', 'Emily_Deschanel', 'Hugh_Dancy', 'Lauren_German', 'Lee_Pace', 'Kate_Hudson', 'Luke_Wilson', 'Sarah_Michelle_Gellar', 'Patrick_Dempsey', 'Ryan_Kwanten', 'Ali_Larter', 'Desmond_Harrington', 'Milla_Jovovich', 'Samantha_Barks', 'Chelsea_Kane', 'Tom_Mison', 'Matthew_Goode', 'Clark_Duke', 'Bryce_Dallas_Howard', 'Rosario_Dawson', 'Eric_Dane', 'Deborah_Ann_Woll', 'Luke_Evans', 'Jason_Segel', 'Elizabeth_Mitchell']

    # woman: 0 man:1
    genders = ['1', '0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '0', '1', '0', '0', '1', '0', '1', '1', '0', '0', '1', '1', '1', '0', '1', '0', '0', '0', '1', '1', '1', '0', '1', '0', '1', '1', '1', '0', '1', '0', '0', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '1', '0', '0', '1', '0', '1', '1', '0', '1', '1', '1', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '1', '1', '0', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '0', '0', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '1', '0', '0', '1', '0', '1', '1', '0', '0', '1', '0', '1', '0', '1', '0', '1', '1', '0', '1', '0', '0', '0', '1', '1', '1', '0', '0', '1', '0', '1', '1', '0']

    # names = ['Kurt_Russell', 'Olivia_Hussey', 'Robin_Williams', 'Mark_Hamill', 'Jane_Seymour', 'Lynda_Carter']
    # genders = ['1', '0', '1', '1', '0', '1']
    index = names.index(file_name)
    return index, int(genders[index])

def get_inner_inter_center_by_age_name(file_names, size_image, image_value_range, num_input_channels, size_age, size_name, size_gender, size_name_total, E_model):

    file_names_by_name_age = []
    age_name_genders = []
    file_names_reorder = []
    inner_centers = []
    inter_centers = []
    age_labels = []
    name_labels = []
    gender_labels = []

    for i, label in enumerate(file_names):

        temp = str(file_names[i]).split('/')[-1]
        # temp = str(file_names[i]).split('\\')[-1]
        age = int(temp.split('_')[0])
        name = temp[temp.index('_') + 1: temp.index('00') - 1]
        age = age_group_label(age)
        [name, gender] = name_gender_label(name)

        age_name_gender = str(age) + '_' + str(name) + '_' + str(gender)
        try:
            index = age_name_genders.index(age_name_gender)
            file_names_by_name_age[index].append(file_names[i])
        except:
            age_name_genders.append(age_name_gender)
            file_names_by_name_age.append([file_names[i]])


    for i in range(len(age_name_genders)):
        # inner centers
        inputs_file_names = file_names_by_name_age[i]
        input_real_images = \
            [load_image(
                image_path=file_name,
                image_size=size_image,
                image_value_range=image_value_range,
                is_gray=(num_input_channels == 1),
        ) for file_name in inputs_file_names]
        if num_input_channels == 1:
            input_real_images = np.array(input_real_images).astype(np.float32)[:, :, :, None]
        else:
            input_real_images = np.array(input_real_images).astype(np.float32)

        num = len(input_real_images)

        age = int(age_name_genders[i].split('_')[0])
        name = int(age_name_genders[i].split('_')[1])
        gender = int(age_name_genders[i].split('_')[-1])

        age_label = np.zeros((num, size_age))
        age_label[:, age] = 1
        name_label = np.zeros((num, size_name))
        if int(size_name) > 1:
            name_label[:, name] = 1
        else:
            name_label[:] = name / size_name_total
        gender_label = np.zeros((num, size_gender))
        gender_label[:, gender] = 1

        # age_label = concat_label(age_label, enable_tile_label, tile_ratio)
        age_label_conv = np.reshape(age_label, [num, 1, 1, age_label.shape[-1]])
        name_label_conv = np.reshape(name_label, [num, 1, 1, name_label.shape[-1]])
        gender_label_conv = np.reshape(gender_label, [num, 1, 1, gender_label.shape[-1]])

        inner_target = E_model.predict([input_real_images, age_label_conv, name_label_conv, gender_label_conv], verbose=0)
        inner_center = copy_array(np.average(inner_target, axis=0), num)

        # inter centers
        current_age_name_genders = age_name_genders[0: i] + age_name_genders[i + 1: len(age_name_genders)]
        for j in range(len(current_age_name_genders)):
            current_age = int(current_age_name_genders[j].split('_')[0])
            current_name = int(current_age_name_genders[j].split('_')[1])
            current_gender = int(current_age_name_genders[j].split('_')[-1])

            if (current_age != age) and (current_name == name):
                index = age_name_genders.index(str(current_age) + '_' + str(current_name) + '_' + str(current_gender))

                current_num = len(file_names_by_name_age[index])
                current_age_label = np.zeros((current_num, size_age))
                current_age_label[:, current_age] = 1
                current_name_label = np.zeros((current_num, size_name))
                if int(size_name) > 1:
                    current_name_label[:, current_name] = 1
                else:
                    current_name_label[:] = current_name / size_name_total
                current_gender_label = np.zeros((current_num, size_gender))
                current_gender_label[:, current_gender] = 1

                # current_age_label = concat_label(current_age_label, enable_tile_label, tile_ratio)
                current_age_label_conv = np.reshape(current_age_label, [current_num, 1, 1, current_age_label.shape[-1]])
                current_name_label_conv = np.reshape(current_name_label, [current_num, 1, 1, current_name_label.shape[-1]])
                current_gender_label_conv = np.reshape(current_gender_label, [current_num, 1, 1, current_gender_label.shape[-1]])

                current_inputs_file_names = file_names_by_name_age[index]
                current_image = [load_image(
                        image_path=file_name,
                        image_size=size_image,
                        image_value_range=image_value_range,
                        is_gray=(num_input_channels == 1),
                ) for file_name in current_inputs_file_names]
                if num_input_channels == 1:
                    current_image = np.array(current_image).astype(np.float32)[:, :, :, None]
                else:
                    current_image = np.array(current_image).astype(np.float32)


                inter_target = E_model.predict([current_image, current_age_label_conv, current_name_label_conv, current_gender_label_conv], verbose=0)
                inter_center = copy_array(-np.average(inter_target, axis=0), num)

        # print('name:',name)
        for ii in range(num):
            file_names_reorder.append(file_names_by_name_age[i][ii])
            inner_centers.append(inner_center[ii].tolist())
            inter_centers.append(inter_center[ii].tolist())
            age_labels.append(age_label[ii].tolist())
            name_labels.append(name_label[ii].tolist())
            gender_labels.append(gender_label[ii].tolist())

    print('Resorted Done!')
    return np.array(file_names_reorder), np.array(inner_centers), np.array(inter_centers), \
           np.array(age_labels), np.array(name_labels), np.array(gender_labels)


def get_age_name_gender_list(file_names, size_batch, size_age, size_name, size_gender, size_name_total):
    age_list, name_list, gender_list = [], [], []
    for i, label in enumerate(file_names):

        temp = str(file_names[i]).split('/')[-1]
        # temp = str(file_names[i]).split('\\')[-1]
        age = int(temp.split('_')[0])
        try:
            name = temp[temp.index('_') + 1: temp.index('n00') - 1]
        except ValueError:
            name = temp[temp.index('_') + 1: temp.index('00') - 1]
        age = age_group_label(age)
        [name, gender] = name_gender_label(name)

        age_label = np.zeros((size_age))
        age_label[age] = 1
        name_label = np.zeros((size_name))
        if int(size_name) > 1:
            name_label[name] = 1
        else:
            name_label = name / size_name_total
        gender_label = np.zeros((size_gender))
        gender_label[gender] = 1

        age_list.append(age_label)
        name_list.append(name_label)
        gender_list.append(gender_label)

    # age_label = np.zeros((size_batch, size_age))
    # age_label[:, age] = 1
    # name_label = np.zeros((size_batch, size_name))
    # if int(size_name) > 1:
    #     name_label[:, name] = 1
    # else:
    #     name_label[:] = name / size_name_total
    # gender_label = np.zeros((size_batch, size_gender))
    # gender_label[:, gender] = 1

    return [np.array(age_list), np.array(name_list), np.array(gender_list)]

def draw_loss_metric(save_dir, npy_name):



    if os.path.exists(save_dir+npy_name + ".jpg"):
        print('remove loss jpg')
        os.remove(save_dir+npy_name + ".jpg")

    # Create a new figure of size 8x6 points, using 80 dots per inch
    # figure(figsize=(8, 6), dpi=80)

    # Create a new subplot from a grid of 1x1
    # subplot(1, 1, 1)

    # X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    # C,S = np.cos(X), np.sin(X)
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    F = np.load(save_dir + npy_name + '.npy')
    if len(F)==0:
        return
    X = np.linspace(1, len(F), len(F), endpoint=True)

    plt.xlim(X.min() * 1.1, X.max() * 1.1)
    min_indx = np.argmin(F)
    plt.plot(min_indx, F[min_indx], 'gs')
    show_min = '[' + str(min_indx) + ' ' + str(F[min_indx]) + ']'
    plt.annotate(show_min, xytext=(min_indx, F[min_indx]), xy=(min_indx, F[min_indx]))


    # # show last point
    # last_indx = len(F) - 1
    # plt.plot(last_indx, F[last_indx], 'gs')
    # show_min = '[' + str(last_indx) + ' ' + str(F[last_indx]) + ']'
    # plt.annotate(show_min, xytext=(last_indx, F[last_indx]), xy=(last_indx, F[last_indx]))
    # # show standard line
    # if npy_name=='loss_Dimg' or npy_name=='loss_GD':
    #     x = np.arange(0,int(last_indx))
    #     y = [0.5]*last_indx
    #     plt.plot(x,y,"r-")

    ax.plot(X, F, color="blue", linewidth=1.0, linestyle="-")
    savefig(save_dir+npy_name + ".jpg")
    ax.clear()
    # if npy_name == 'loss_all':
    #
    #     temp = F * np.sum(loss_weights)
    #
    #     if loss_weights[0] != 0:
    #         plt.xlim(X.min() * 1.1, X.max() * 1.1)
    #         loss_E = temp/ loss_weights[0]
    #         min_indx = np.argmin(loss_E)
    #         plt.plot(min_indx, loss_E[min_indx], 'gs')
    #         show_min = '[' + str(min_indx) + ' ' + str(loss_E[min_indx]) + ']'
    #         plt.annotate(show_min, xytext=(min_indx, loss_E[min_indx]), xy=(min_indx, loss_E[min_indx]))
    #         ax.plot(X, loss_E, color="blue", linewidth=1.0, linestyle="-" )
    #         savefig(save_dir + 'loss_E(loss all)' + ".jpg")
    #         ax.clear()
    #
    #     if loss_weights[1] != 0 and loss_weights[2] != 0:
    #         plt.xlim(X.min() * 1.1, X.max() * 1.1)
    #         loss_Dimg = temp/ loss_weights[1] + temp/loss_weights[2]
    #         min_indx = np.argmin(loss_Dimg)
    #         plt.plot(min_indx, loss_Dimg[min_indx], 'gs')
    #         show_min = '[' + str(min_indx) + ' ' + str(loss_Dimg[min_indx]) + ']'
    #         plt.annotate(show_min, xytext=(min_indx, loss_Dimg[min_indx]), xy=(min_indx, loss_Dimg[min_indx]))
    #         ax.plot(X, loss_Dimg, color="blue", linewidth=1.0, linestyle="-" )
    #         savefig(save_dir + 'loss_Dimg(loss all)' + ".jpg")
    #         ax.clear()
    #
    #     if loss_weights[3] != 0:
    #         plt.xlim(X.min() * 1.1, X.max() * 1.1)
    #         loss_GD = temp/ loss_weights[3]
    #         min_indx = np.argmin(loss_GD)
    #         plt.plot(min_indx, loss_GD[min_indx], 'gs')
    #         show_min = '[' + str(min_indx) + ' ' + str(loss_GD[min_indx]) + ']'
    #         plt.annotate(show_min, xytext=(min_indx, loss_GD[min_indx]), xy=(min_indx, loss_GD[min_indx]))
    #         ax.plot(X, loss_GD, color="blue", linewidth=1.0, linestyle="-" )
    #         savefig(save_dir + 'loss_GD(loss all)' + ".jpg")
    #         ax.clear()
    #
    #     if loss_weights[4] != 0:
    #         plt.xlim(X.min() * 1.1, X.max() * 1.1)
    #         loss_image = temp/ loss_weights[4]
    #         min_indx = np.argmin(loss_image)
    #         plt.plot(min_indx, loss_image[min_indx], 'gs')
    #         show_min = '[' + str(min_indx) + ' ' + str(loss_image[min_indx]) + ']'
    #         plt.annotate(show_min, xytext=(min_indx, loss_image[min_indx]), xy=(min_indx, loss_image[min_indx]))
    #         ax.plot(X, loss_image, color="blue", linewidth=1.0, linestyle="-" )
    #         savefig(save_dir + 'loss_image(loss all)' + ".jpg")
    #         ax.clear()

    # show()

def preprocess_image(images, size_image, num_input_channels, size_scale):
    # from keras import backend as K

    inputs = keras.backend.placeholder(shape=(None, size_image, size_image, num_input_channels))
    seed = np.random.randint(0, 2 ** 31 - 1)
    inputs = tf.image.resize_images(inputs, [size_scale, size_scale], method=tf.image.ResizeMethod.AREA)
    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, size_scale - size_image + 1, seed=seed)),
                     dtype=tf.int32)
    inputs = tf.image.crop_to_bounding_box(inputs, offset[0], offset[1], size_image, size_image)

    with tf.Session() as sess:
        images = sess.run(inputs, feed_dict={inputs: images})

    return images

def get_intermediate_output(model, layer_name, inputs):
    from keras.models import Model
    intermediate_layer_model = Model(input=model.inputs, output=model.get_layer(name=layer_name).output)
    intermediate_output = intermediate_layer_model.predict(inputs)
    return intermediate_output[0]

def image_resize_from_array(array_expand_dim, image_resize):
    from keras.preprocessing import image as kimage
    image_list = []
    (num, height, width, dim) = array_expand_dim.shape
    for i in range(num):
        array = array_expand_dim[i]
        array = (array + 1)/2 + 255.0
        image = kimage.array_to_img(array)
        image = imresize(image, [image_resize, image_resize])
        image = image.astype(np.float32) * 2 / 255.0 - 1
        image_list.append(image)
    return np.array(image_list)

def sorted_compare_image_pairs(real_images, file_names, size_age, size_name, size_gender, size_name_total):
    compare_images_pick, compare_ages_pick, real_images_pick = [], [], []

    [real_ages, real_names, real_genders] = get_age_name_gender_list(file_names, len(file_names), size_age, size_name, size_gender, size_name_total)


    for i in range(len(real_ages)):
        real_age, real_name, real_gender = real_ages[i], real_names[i], real_genders[i]
        compare_images, compare_ages = [], []
        for j in range(len(real_ages)):
            if recover_age(real_ages[j]) != recover_age(real_age) and recover_name(real_names[j]) == recover_name(real_name):
                compare_images.append(real_images[j])
                compare_ages.append(real_ages[j])

        count = len(compare_images)
        if count > 0:
            random_index = np.random.randint(0, count, 1)
            print('--------------------randim index----------------' + str(random_index[0]))
            compare_images_pick.append(compare_images[random_index[0]])
            compare_ages_pick.append(compare_ages[random_index[0]])
            real_images_pick.append(real_images[i])

    return [np.array(real_images_pick), np.array(compare_images_pick), np.array(real_ages), np.array(compare_ages_pick), np.array(real_names), np.array(real_genders)]

def recover_age(age_label):
    for i in range(len(age_label)):
        if int(age_label[i]) == 1:
            return i

def recover_name(name_label):
    if len(name_label) == 1:
        return name_label[0]
    elif len(name_label) > 1:
        for i in range(len(name_label)):
            if int(name_label[i]) == 1:
                return i

if __name__ == '__main__':
    from glob import glob

    file_names = glob(os.path.join('./data/new_128', '*.jpg'))
    # real_images = [load_sample(file_name, 128, image_value_range=(-1, 1), is_gray=False) for file_name in file_names]
    # sorted_compare_image_pairs(real_images, file_names, 10, 6, 2, 6)
    load_image(file_names, 128, 158, 3)