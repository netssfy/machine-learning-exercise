#predict if my little pomelo is existence in a photo or not

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNEL = 3

CONV1_KERNEL_SIZE = [8, 8]
CONV1_FILTER = 16
POOL1_SIZE = [4, 4]

CONV2_KERNEL_SIZE = [6, 6]
CONV2_FILTER = 32
POOL2_SIZE = [4, 4]

CONV3_KERNEL_SIZE = [8, 8]
CONV3_FILTER = 128
POOL3_SIZE = [2, 2]

DENSE_UNITS = 1024

home_dir = 'D:\\Projects\\machine-learning-exercise\\deep-learning\\pomelo-predictor'
image_dir = home_dir + '\\pomelos\\'

def cnn_model_fn(features, labels, mode):
    print('input layer [{}, {}, {}, {}]'.format(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL))
    input_layer = tf.reshape(features['x'], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL])
    print('actual {}'.format(input_layer.shape))

    print('conv1 output [{}, {}, {}, {}]'.format(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CONV1_FILTER))
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=CONV1_FILTER,
        kernel_size=CONV1_KERNEL_SIZE,
        padding='same',
        activation=tf.nn.relu)
    print('actual {}'.format(conv1.shape))

    print('pool1 output [{}, {}, {}, {}]'.format(-1, IMAGE_WIDTH / POOL1_SIZE[0], IMAGE_HEIGHT / POOL1_SIZE[1], CONV1_FILTER))
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=POOL1_SIZE,
        strides=POOL1_SIZE[0])
    print('actual {}'.format(pool1.shape))

    print('conv2 output [{}, {}, {}, {}]'.format(-1, IMAGE_WIDTH / POOL1_SIZE[0], IMAGE_HEIGHT / POOL1_SIZE[1], CONV2_FILTER))
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=CONV2_FILTER,
        kernel_size=CONV2_KERNEL_SIZE,
        padding='same',
        activation=tf.nn.relu)
    print('actual {}'.format(conv2.shape))

    print('pool2 output [{}, {}, {}, {}]'.format(-1, IMAGE_WIDTH / POOL1_SIZE[0] / POOL2_SIZE[0], IMAGE_HEIGHT / POOL1_SIZE[1] / POOL2_SIZE[1], CONV2_FILTER))
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=POOL2_SIZE,
        strides=POOL2_SIZE[0])
    print('actual {}'.format(pool2.shape))
    '''
    #[*, 16, 16, 128]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=CONV3_FILTER,
        kernel_size=CONV3_KERNEL_SIZE,
        padding='same',
        activation=tf.nn.relu)

    #[*, 8, 8, 128]
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=POOL3_SIZE,
        strides=2)
    '''

    flat = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
    dense = tf.layers.dense(inputs=flat, units=DENSE_UNITS, activation=tf.nn.relu)
    print('dense actual {}'.format(dense.shape))

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = { 
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
        }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    data = load_pomelo()

    pomelo_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=os.getcwd() + '/tmp/pomelo_model')
    tensors_to_log = {'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=25)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': data['train_x']},
        y=data['train_y'],
        batch_size=50,
        num_epochs=None,
        shuffle=False)

    startTick = time.time()
    print('begin training at {}...'.format(startTick))
    pomelo_classifier.train(input_fn=train_input_fn, steps=200, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': data['eval_x']},
        y=data['eval_y'],
        num_epochs=1,
        shuffle=False)

    print('begin eval...')
    eval_results = pomelo_classifier.evaluate(input_fn=eval_input_fn)


    print('I am done. {} ticks elapsed'.format(time.time() - startTick))
    print(eval_results)

def load_pomelo():
    #mnist = tf.contrib.learn.datasets.load_dataset('mnist');
    #return { 'x': mnist.train.images, 'y': np.asarray(mnist.train.labels, dtype=np.int32) }
    train_x = []
    train_y = []
    eval_x = []
    eval_y = []
    dataset = np.genfromtxt(image_dir + 'y.csv', delimiter=',', dtype='U8,i4')
    index = 0
    with tf.Session():
        for row in dataset:
            index = index + 1
            fn = row[0]
            image = tf.gfile.FastGFile(image_dir + fn, 'rb').read()
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize_images(image, [IMAGE_WIDTH, IMAGE_HEIGHT], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if index <= 50:
                train_x.append(processImage(image))
                train_y.append(row[1])
            else:
                eval_x.append(processImage(image))
                eval_y.append(row[1])

    return { 
        'train_x': np.array(train_x),
        'train_y': np.array(train_y),
        'eval_x': np.array(eval_x),
        'eval_y': np.array(eval_y)
    }

def processImage(image):
    imageArray = np.array(image.eval()).flatten()
    return np.divide(imageArray, np.full(imageArray.shape, 255, dtype='float16'))
    

main({})