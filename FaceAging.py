# -*- coding: UTF-8 -*
import generator, gan, discriminator, loss, content_vgg
import tensorflow as tf
from ops import load_image, age_group_label, duplicate, \
    concat_label, save_image, save_weights, save_loss, \
    draw_loss_metric, get_age_name_gender_list, preprocess_image, \
    load_sample, get_intermediate_output, image_resize_from_array, sorted_compare_image_pairs
from keras.optimizers import SGD, Adam
from keras.models import load_model
from glob import glob
import os
import numpy as np
from keras.utils import multi_gpu_model
from keras import backend as K
from pylab import plt
import sys



class FaceAging(object):

    def __init__(self,
                 size_image,  # size the input images
                 size_scale,  # size the scale
                 size_batch,  # size of one batch
                 dataset_name,  # name of the dataset in the folder ./data
                 sample_dir,

                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 num_input_channels=3,  # number of channels of input images

                 size_z=100,  # number of channels of the layer z (noise or code)
                 # num_encoder_channels=[64, 128, 256, 512],  # number of channels of every conv layers of encoder
                 # num_encoder_channels=[512, 256, 128, 64, 3],  # number of channels of every conv layers of encoder
                 num_encoder_channels=[64, 128, 256, 512, 512, 512, 512],  # number of channels of every conv layers of encoder

                 size_gen=512, # number of channels of the generator's start layer
                 # num_gen_channels=[512, 256, 128, 64, 3],  # number of channels of every deconv layers of generator
                 num_gen_channels=[(512, 0), (512, 0), (512, 0), (256, 0), (128, 0), (64, 0), (3, 0)],  # number of channels of every deconv layers of generator

                 num_Dz_channels=[128, 64, 32, 16, 1],  # number of channels of every conv layers of discriminator_z

                 #num_Dimg_channels=3,  # number of channels of discriminator input image
                 num_Dimg_channels=[32, 64, 64*2, 64*4, 64*8],  #number of channels of  every conv layers of discriminator_img
                 num_Dimg_fc_channels = 1024, # number of channels of last fc layer of discriminator_img

                 # content_channels
                 content_channels=['block1_conv1', 'block1_conv2', 'block1_pool',
                                   'block2_conv1', 'block2_conv2', 'block2_pool',
                                   'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool',
                                   'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool',
                                   'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'],
                 num_content_channels=[(128, 128, 3), (128, 128, 3), (64, 64, 3),
                                       (64, 64, 3), (64, 64, 3), (32, 32, 3),
                                       (32, 32, 3), (32, 32, 3), (32, 32, 3), (32, 32, 3), (16, 16, 3),
                                       (16, 16, 3), (16, 16, 3), (16, 16, 3), (16, 16, 3), (8, 8, 3),
                                       (8, 8, 3), (8, 8, 3), (8, 8, 3), (8, 8, 3), (4, 4, 3)],
                 content_channels_pick=['block1_conv2', 'block3_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4'],

                 size_age=10,  # number of categories (age segments) in the training dataset
                 size_name=133, # number of name array
                 size_name_total=133, # number of total name
                 size_gender=2, # male & female

                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary

                 image_mode='RGB',
                 loss_weights=[0, 0, 0, 0],
                 num_GPU=2,
                 num_D_img_loss=1,
                 num_all_loss=1,
                 GANs='cGAN',
                 G_net='Unet'
                 ):
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_scale = size_scale
        self.size_batch = size_batch
        self.size_kernel = size_kernel
        self.num_input_channels = num_input_channels
        self.size_z = size_z
        self.num_encoder_channels = num_encoder_channels
        self.num_Dz_channels = num_Dz_channels
        self.num_Dimg_channels = num_Dimg_channels
        self.num_Dimg_fc_channels = num_Dimg_fc_channels
        self.content_channels = content_channels
        self.num_content_channels = num_content_channels
        self.content_channels_pick = content_channels_pick
        self.size_age = size_age
        self.size_name = size_name
        self.size_name_total = size_name_total
        self.size_gender = size_gender
        self.size_gen = size_gen
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name
        self.sample_dir = sample_dir
        self.image_mode = image_mode
        # self.loss_weights = [float(i) for i in loss_weights/np.sum(loss_weights)]
        self.loss_weights = loss_weights
        self.num_GPU = num_GPU
        self.num_D_img_loss = num_D_img_loss
        self.num_all_loss = num_all_loss
        self.GANs = GANs
        self.G_net = G_net

        print("\n\tBuilding the graph...")

        # label of age + gender duplicate size
        self.size_age_label = duplicate(
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            size_age=self.size_age)

        # generator model: z + label --> generated image
        self.G_model = generator.generator_pix2pix_model(
            size_image=self.size_image,
            size_age_label=self.size_age_label,
            size_name_label=self.size_name,
            size_gender_label=self.size_gender,
            size_kernel=self.size_kernel,
            num_input_channels=self.num_input_channels,
            num_encoder_channels=self.num_encoder_channels,
            num_gen_channels=self.num_gen_channels,
            G_net=self.G_net
        )

        # discriminator model on G
        self.D_img_model = discriminator.discriminator_img_model(
            size_image=self.size_image,
            size_kernel=self.size_kernel,
            size_age_label=self.size_age_label,
            size_name_label=self.size_name,
            size_gender_label=self.size_gender,
            num_input_channels=self.num_input_channels,
            num_Dimg_channels=self.num_Dimg_channels,
            num_Dimg_fc_channels=self.num_Dimg_fc_channels,
            GANs=self.GANs
        )

        self.content_model = content_vgg.content_model(
            size_image=self.size_image,
            num_input_channels=self.num_input_channels,
            content_channels_pick=self.content_channels_pick
        )

        # # G + D_img Model
        self.GD_model = gan.gdModel(
            self.G_model, self.D_img_model,
            self.size_image, self.size_age_label, self.size_name, self.size_gender, self.num_input_channels)


        # ****************************** multi-GPU ***********************************
        # self.G_model = multi_gpu_model(self.G_model, gpus=self.num_GPU)
        # self.D_img_model = multi_gpu_model(self.D_img_model, gpus=self.num_GPU)
        # self.GD_model = multi_gpu_model(self.GD_model, gpus=self.num_GPU)

        # ****************************** learning_rate ***********************************
        learning_rate = 0.0002
        decay_rate = 1.0
        global_step = 0
        all_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=516,
            decay_rate=decay_rate,
            staircase=True
        )

        # ************************************* optimizer *******************************************
        adam_G = Adam(lr=all_learning_rate, beta_1=0.5)
        adam_GD = Adam(lr=all_learning_rate, beta_1=0.5)
        adam_D_img = Adam(lr=all_learning_rate, beta_1=0.5)
        adam_loss = Adam(lr=all_learning_rate, beta_1=0.5, beta_2=0.99)

        # ************************************* Compile loss  *******************************************************
        # ************************** EGD weighted loss *************************************
        # ******************* seperate model **********************

        # first calculate content_layer_dim
        num_content_channels_pick = []
        for content in self.content_channels_pick:
            index = self.content_channels.index(content)
            num_content_channels_pick.append(self.num_content_channels[index])

        if self.loss_weights[0] == 0:
            # loss model of Generator MSE/MAE loss(G_loss) and GD_loss
            self.loss_Model = loss.loss_seperate_model(self.size_batch,
                self.size_image, self.size_age, self.size_name, self.size_gender, self.num_input_channels, num_content_channels_pick,
                self.G_model, self.D_img_model, self.content_model, self.GANs)
            self.loss_Model.compile(optimizer=adam_loss, loss=lambda y_true, y_pred: y_pred,
                                    loss_weights=[self.loss_weights[2], self.loss_weights[3], self.loss_weights[4]])


            if self.GANs == 'cGAN':
                # loss model of discriminator on generated + real image
                self.D_img_model.trainable = True
                self.D_img_model.compile(optimizer=adam_D_img, loss='binary_crossentropy')
            elif self.GANs == 'LSGAN':
                # LSGAN argument: a=c=1 ，b=0
                # loss model of discriminator on generated + real image

                def mse_loss_LSGAN(y_true, y_pred):
                    loss_val = tf.reduce_mean(tf.sqrt(2 * tf.nn.l2_loss(y_true - y_pred))/ self.size_batch)
                    return loss_val

                self.GD_model.compile(optimizer=adam_GD, loss='mse')
                self.D_img_model.trainable = True
                self.D_img_model.compile(optimizer=adam_D_img, loss='mse')

        # ************************** all model **********************
        else:
            self.loss_Model = loss.loss_all_model(
                self.size_image, self.size_age, self.size_name, self.size_gender, self.num_input_channels,
                self.G_model, self.D_img_model, self.GANs)
            self.loss_Model.compile(optimizer=adam_loss, loss=lambda y_true, y_pred: y_pred, loss_weights=self.loss_weights)





    def train(self,
              num_epochs,  # number of epochs
              size_batch,  # mini-batch size for training and testing, must be square of an integer
              path_data,  # upper dir of data
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              ):

        #  *************************** load file names of images ***************************************
        file_names = glob(os.path.join(path_data, self.dataset_name, '*.jpg'))
        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')
        sess = K.get_session()

        print('\n\tPreprocessing the dataset ...')
        file_images = load_image(file_names, self.size_image, self.size_scale, self.num_input_channels)
        file_images = np.array(sess.run(file_images))
        [real_images, compare_images, real_ages, compare_ages, real_names, real_genders] = \
            sorted_compare_image_pairs(file_images, file_names, self.size_age, self.size_name, self.size_gender, self.size_name_total)

        # load model
        if use_trained_model:
            if os.path.exists('GD.h5') and os.path.exists('content.h5'):
                self.GD_model.load_weights('GD.h5')
                self.content_model.load_weights('content.h5')
                print("\t**************** LOADING MODELs SUCCESS! ****************")
            else:
                print("\t**************** LOADING MODELs FAILED! ****************")

        # *************************** preparing data for epoch iteration ************************************
        loss_Dimg, loss_D, loss_GD, loss_content, loss_image, loss_all = [], [], [], [], [], []

        for epoch in range(num_epochs):
            num_batches = len(real_images) // size_batch
            for index_batch in range(num_batches):
                # ********************** real batch images and labels **********************
                # real images
                batch_real_images = real_images[index_batch * size_batch:(index_batch + 1) * size_batch]
                # compare images (same person with different age_label pair_ratio=1)
                batch_compare_images = compare_images[index_batch * size_batch:(index_batch + 1) * size_batch]

                batch_real_ages = real_ages[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_compare_ages = compare_ages[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_real_names = real_names[index_batch * size_batch:(index_batch + 1) * size_batch]
                batch_real_genders = real_genders[index_batch * size_batch:(index_batch + 1) * size_batch]

                batch_real_ages_conv = np.reshape(batch_real_ages, [len(batch_real_ages), 1, 1, batch_real_ages.shape[-1]])
                batch_compare_ages_conv = np.reshape(batch_compare_ages, [len(batch_compare_ages), 1, 1, batch_compare_ages.shape[-1]])
                batch_real_names_conv = np.reshape(batch_real_names, [len(batch_real_names), 1, 1, batch_real_names.shape[-1]])
                batch_real_genders_conv = np.reshape(batch_real_genders, [len(batch_real_genders), 1, 1, batch_real_genders.shape[-1]])



                # batch_random = np.random.uniform(self.image_value_range[0], self.image_value_range[1], size=batch_real_images.shape)
                # batch_fake_images = self.G_model.predict([batch_random, batch_compare_ages_conv, batch_real_names_conv, batch_real_genders_conv])
                batch_fake_images = self.G_model.predict([batch_real_images, batch_compare_ages_conv, batch_real_names_conv, batch_real_genders_conv])
                num_D_last_channel = int(self.size_image / 2 ** (len(self.num_Dimg_channels) - 1)) # D_img_model first&last layer strides=1

                from scipy.misc import imsave
                imsave(self.save_dir + '/compare/e'+ str(epoch)+'_b'+str(index_batch)+'_r.png',(batch_real_images[0]+1)/2)
                imsave(self.save_dir + '/compare/e' + str(epoch) + '_b' + str(index_batch) + '_f.png',(batch_fake_images[0] + 1) / 2)




                for all in range(int(self.num_all_loss)):
                    self.D_img_model.trainable = False

                    if self.loss_weights[0] != 0: # train all loss
                        [loss_all_batch, loss_D_fake_batch, loss_D_real_batch, loss_GD_batch, loss_image_batch] = \
                            self.loss_Model.train_on_batch([batch_real_images, batch_fake_images,
                                      batch_compare_ages_conv, batch_real_names_conv, batch_real_genders_conv],
                            [np.zeros(self.size_batch), np.zeros(self.size_batch), np.zeros(self.size_batch), np.zeros(self.size_batch)])
                        loss_D.append(loss_D_fake_batch + loss_D_real_batch)
                        loss_GD.append(loss_GD_batch)
                        loss_image.append(loss_image_batch)
                        loss_all.append(loss_all_batch)
                    else: # train seperated loss
                        [loss_all_batch, loss_GD_batch, loss_content_batch, loss_image_batch] = \
                            self.loss_Model.train_on_batch([batch_compare_images, batch_fake_images,
                                 batch_compare_ages_conv, batch_real_names_conv, batch_real_genders_conv],
                                [np.zeros(self.size_batch), np.zeros(self.size_batch), np.zeros(self.size_batch)])
                        # loss_GD.append(loss_GD_batch)
                        loss_content.append(loss_content_batch)
                        loss_image.append(loss_image_batch)
                        loss_all.append(loss_all_batch)
                    print('loss_all on b_', index_batch, 'e_', epoch, ' is', loss_all[-1])

                    self.D_img_model.trainable = True
                    # ************************* end train EGD_model with all loss ****************************



                for D in range(int(self.num_D_img_loss)):
                    # ************************* start train D_img_model with D_img_loss ****************************
                    train_batch_x = np.concatenate((batch_compare_images, batch_fake_images), axis=0)
                    train_batch_ages_conv = np.concatenate((batch_compare_ages_conv, batch_compare_ages_conv), axis=0)
                    train_batch_names_conv = np.concatenate((batch_real_names_conv, batch_real_names_conv,), axis=0)
                    train_batch_genders_conv = np.concatenate((batch_real_genders_conv, batch_real_genders_conv), axis=0)
                    # label
                    train_batch_y = np.concatenate((np.ones(self.size_batch), np.zeros(self.size_batch)), axis=0)
                    # label ----> Patch D
                    # A = np.ones((size_batch, num_D_last_channel, num_D_last_channel, 1))
                    # B = np.zeros((size_batch, num_D_last_channel, num_D_last_channel, 1))
                    # train_batch_y = np.concatenate((A, B), axis=0)

                    # train D_img
                    loss_Dimg.append(
                        self.D_img_model.train_on_batch([train_batch_x, train_batch_ages_conv, train_batch_names_conv, train_batch_genders_conv], train_batch_y))
                    loss_GD.append(self.GD_model.train_on_batch([batch_fake_images, batch_compare_ages_conv, batch_real_names_conv, batch_real_genders_conv],
                                                                np.ones(self.size_batch)))
                    print('loss_Dimg on b_', index_batch, ' e_', epoch, ' is ', loss_Dimg[-1])
                    print('loss_GD on b_', index_batch, ' e_', epoch, ' is ', loss_GD[-1])
                    # ************************* end train D_img_model with D_img_loss ****************************


                # *********************************** save images *******************************************
                # *************************  start E_model to generate latant_z ***********************************
                if (epoch % 5 == 0) or (epoch == 1):
                # if (index_batch % 100 == 0) or (index_batch == 1):
                    sample_names = glob(os.path.join(path_data, self.sample_dir, '*.jpg'))
                    sample_images = [load_sample(
                        image_path=file_name,
                        image_size=self.size_image,
                        image_value_range=self.image_value_range,
                        is_gray=(self.num_input_channels == 1),
                    ) for file_name in sample_names]
                    if self.num_input_channels == 1:
                        sample_images = np.array(sample_images).astype(np.float32)[:, :, :, None]
                    else:
                        sample_images = np.array(sample_images).astype(np.float32)


                    num_sample = len(sample_images)
                    fake_age = np.zeros((num_sample, self.size_age))
                    fake_age[:, 6] = 1
                    fake_age_conv = np.reshape(fake_age, [len(fake_age), 1, 1, fake_age.shape[-1]])
                    fake_name = np.zeros((num_sample, self.size_name))
                    if int(self.size_name) > 1:
                        fake_name[:, 0] = 1
                    else:
                        fake_name[:] = 1 / self.size_name_total
                    fake_name_conv = np.reshape(fake_name, [len(fake_name), 1, 1, fake_name.shape[-1]])
                    fake_gender = np.zeros((num_sample, self.size_gender))
                    fake_gender[:, 1] = 1
                    fake_gender_conv = np.reshape(fake_gender, [len(fake_gender), 1, 1, fake_gender.shape[-1]])

                    sample_fake_image = self.G_model.predict([sample_images, fake_age_conv, fake_name_conv, fake_gender_conv])
                    save_image(sample_fake_image, self.size_image, self.image_value_range, self.num_input_channels, epoch, index_batch, self.image_mode,
                               self.save_dir + '/image')

                # ************************  end of E to generate latant_z ************************

            if (epoch % 20 == 0) or (epoch == 1):
                save_weights(self.save_dir + '/weight', self.GD_model, self.content_model, epoch, 0)
                save_loss(self.save_dir+'/metric', loss_Dimg, loss_D, loss_GD, loss_content, loss_image, loss_all)
                if len(loss_Dimg) != 0:
                    draw_loss_metric(self.save_dir + '/metric/', 'loss_Dimg')
                else:
                    draw_loss_metric(self.save_dir + '/metric/', 'loss_Dimg(all)')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_GD')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_content')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_image')
                draw_loss_metric(self.save_dir + '/metric/', 'loss_all')


