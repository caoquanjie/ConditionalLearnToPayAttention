from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import skimage
import skimage.transform
import skimage.io

def _parameter_summary(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.reduce_sum(tf.pow(tf.norm(params, axis=(0,1)), 2), axis=1))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _parameter_summary_fc(params):
    tf.summary.histogram(params.op.name, params)
    tf.summary.histogram(params.op.name + '/row_norm', tf.pow(tf.norm(params, axis=1), 2))
    tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _output_summary(outputs):
    tf.summary.histogram(outputs.op.name + '/outputs', outputs)
    tf.summary.scalar(outputs.op.name + '/outputs_sparsity',
		tf.nn.zero_fraction(outputs))


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name=name)


def conv_layer_BN(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        n = w_shape[0]*w_shape[1]*w_shape[3]
        filt = get_conv_filter(name, w_shape,n)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        _parameter_summary_fc(filt)
        _output_summary(bias)

    return bias

def relu_layer(x,name):
    with tf.variable_scope(name):
        h = tf.nn.relu(x,name=name)
        _output_summary(h)

    return h



def batch_normalization_layer(x,axis,phase,name):
    with tf.variable_scope(name):
        h = tf.layers.batch_normalization(x,axis=axis,training=phase,name=name)
    return h

def dropout_layer(x,drop_rate,is_train):
    return tf.layers.dropout(inputs=x,rate=drop_rate,training=is_train)

def ConvBNReLU(bottom,w_shape,b_shape,axis,phase,name):
    with tf.variable_scope(name):
        h = conv_layer_BN(bottom,name=name+'/conv',w_shape=w_shape,b_shape=b_shape)

    with tf.variable_scope(name):
        h = batch_normalization_layer(h,axis=axis,phase=phase,name=name+'/BN')

    with tf.variable_scope(name):
        h = relu_layer(h,name = name+'/ReLU')

    return h



def decoder_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_decoder_weight(name, w_shape)
        biases = get_decoder_biases(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)

        return fc


def fc_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name, w_shape)
        biases = get_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)

        return fc


def get_decoder_weight(name,shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.01),name)

def get_decoder_biases(name,shape):
    return tf.Variable(tf.constant(0.0,shape=shape),name=name)


def get_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_conv_filter(name, shape,n):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_bias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_trans_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def get_trans_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())

def trans_layer(bottom, name, w_shape, b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_trans_fc_weight(name, w_shape)
        biases = get_trans_fcbias(name, b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        _parameter_summary_fc(weights)
        _output_summary(fc)
        return fc


def compatibility_func(L,g):
    size = L.get_shape().as_list()[1]
    batch = L.get_shape().as_list()[0]
    L_vector = tf.reshape(L,[batch,size*size,-1]) #(batch,56*56,256)
    L_vector_T = tf.transpose(L_vector, (0, 2, 1))
    g_vector = tf.expand_dims(g,axis=2)
    score_vector = tf.reduce_sum(tf.multiply(L_vector_T,g_vector),1) #(batch,56*56)
    score = tf.nn.softmax(score_vector) #(batch,56*56)
    imgs = tf.expand_dims(score,axis=1)  #(batch,1,56*56)
    a_score = tf.expand_dims(score,axis=1) #(batch,1,56*56)
    gas = tf.multiply(L_vector_T, a_score)  # (batch,256,56*56)
    ga = tf.reduce_sum(gas,[2])
    print(ga.shape)
    print(imgs.shape)
    return ga,imgs



def save_p3_train_results(images,labels,feats3,config,step,predict,probs):
    if not os.path.exists(config.train_p3_result_dir):
        os.mkdir(config.train_p3_result_dir)
    fig = plt.figure(1, figsize=(10, 10))

    plt.subplot(1, len(feats3) + 1, 1)
    plt.title(labels[0,:5])
    plt.imshow(images[0])

    for idx in range(len(feats3)):
        plt.subplot(1,len(feats3)+1,idx+2)
        plt.title("{}({:3.4f})".format(predict[idx][0], probs[idx][0]))
        plt.imshow(images[0])
        img = skimage.transform.pyramid_expand(feats3[idx][0, 0, :].reshape(14, 14), upscale=4,sigma=10)
        plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.train_p3_result_dir, step), bbox_inches='tight')

def save_p2_train_results(images, labels,feats2, config, step, predict, probs):
    if not os.path.exists(config.train_p2_result_dir):
        os.mkdir(config.train_p2_result_dir)
    fig = plt.figure(1, figsize=(10, 10))

    plt.subplot(1, len(feats2) + 1, 1)
    plt.title(labels[0, :5])
    plt.imshow(images[0])

    for idx in range(len(feats2)):
        plt.subplot(1, len(feats2) + 1, idx + 2)
        plt.title("{}({:3.4f})".format(predict[idx][0], probs[idx][0]))
        plt.imshow(images[0])
        img = skimage.transform.pyramid_expand(feats2[idx][0, 0, :].reshape(27, 27), upscale=2, sigma=10)
        plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.train_p2_result_dir, step), bbox_inches='tight')

def save_p3_test_results(images,labels,feats3,config,step,predict,probs):
    if not os.path.exists(config.test_p3_result_dir):
        os.mkdir(config.test_p3_result_dir)
    fig = plt.figure(1, figsize=(10, 10))

    plt.subplot(1, len(feats3) + 1, 1)
    plt.title(labels[0,:5])
    plt.imshow(images[0])

    for idx in range(len(feats3)):
        plt.subplot(1,len(feats3)+1,idx+2)
        plt.title("{}({:3.4f})".format(predict[idx][0], probs[idx][0]))
        plt.imshow(images[0])
        img = skimage.transform.pyramid_expand(feats3[idx][0, 0, :].reshape(14, 14), upscale=4,sigma=10)
        plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.test_p3_result_dir, step), bbox_inches='tight')


def save_p2_test_results(images, labels,feats2, config, step, predict, probs):
    if not os.path.exists(config.test_p2_result_dir):
        os.mkdir(config.test_p2_result_dir)
    fig = plt.figure(1, figsize=(10, 10))

    plt.subplot(1, len(feats2) + 1, 1)
    plt.title(labels[0, :5])
    plt.imshow(images[0])

    for idx in range(len(feats2)):
        plt.subplot(1, len(feats2) + 1, idx + 2)
        plt.title("{}({:3.4f})".format(predict[idx][0], probs[idx][0]))
        plt.imshow(images[0])
        img = skimage.transform.pyramid_expand(feats2[idx][0, 0, :].reshape(27, 27), upscale=2, sigma=10)
        plt.imshow(img, alpha=0.8)

    plt.savefig('{}/{}.png'.format(config.test_p2_result_dir, step), bbox_inches='tight')
