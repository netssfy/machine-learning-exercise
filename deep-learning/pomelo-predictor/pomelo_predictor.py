#predict if my little pomelo is existence in a photo or not

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

home_dir = 'D:\\Projects\\machine-learning-exercise\\deep-learning\\pomelo-predictor'

image = tf.gfile.FastGFile(home_dir + '\\pomelos\\1.png', 'rb').read()

with tf.Session() as sess:
    tr_image = tf.image.decode_png(image)
    tr_image = tf.image.resize_images(tr_image, [128, 128])
    tr_image = tf.div(tr_image, tf.constant(255.0, shape = [128, 128, 4]))
    plt.imshow(tr_image.eval())

    tr_image_flip = tf.image.flip_left_right(tr_image)

    tmp = tf.image.encode_png(tr_image)
    tf.gfile.FastGFile(home_dir + '\\pomelos\\2.png', 'wb').write(tmp.eval())
    