#predict if my little pomelo is existence in a photo or not

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

home_dir = 'D:\\Projects\\machine-learning-exercise\\deep-learning\\pomelo-predictor'

image = tf.gfile.FastGFile(home_dir + '\\pomelos\\1.png', 'rb').read()

with tf.Session() as sess:
    tr_image = tf.image.decode_png(image)
    tr_image = tf.image.resize_images(tr_image, [128, 128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #plt.imshow(tr_image.eval())

    tr_image_flip = tf.image.flip_left_right(tr_image)

'''
    tmp = tf.image.encode_png(tr_image)
    tf.gfile.FastGFile(home_dir + '\\pomelos\\2.png', 'wb').write(tmp.eval())
    tmp = tf.image.encode_png(tr_image_flip)
    tf.gfile.FastGFile(home_dir + '\\pomelos\\3.png', 'wb').write(tmp.eval())
'''

def cnn_model_fn(features, labels, mode):
    #[*, 128, 128, 3]
    input_layer = tf.reshape(features, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    #[*, 128, 128, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[4, 4],
        padding='same',
        activation=tf.nn.relu)

    #[*, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    #[*, 64, 64, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[6, 6],
        padding='same',
        activation=tf.nn.relu)

    #[*, 16, 16, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[4, 4],
        strides=4)

    #[*, 16, 16, 128]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[8, 8],
        padding='same',
        activation=tf.nn.relu)

    #[*, 8, 8, 128]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=2)

