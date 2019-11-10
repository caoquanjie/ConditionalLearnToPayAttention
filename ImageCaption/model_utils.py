from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


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


def get_fcbias(name, shape):
    return tf.get_variable(name + '_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(0))


def get_fc_weight(name, shape):
    return tf.get_variable(name + '_W', dtype=tf.float32, shape=shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))


def compatibility_func(L,g):
    size = L.get_shape().as_list()[1]
    batch = L.get_shape().as_list()[0]
    L_vector = tf.reshape(L,[batch,size*size,-1]) #(batch,56*56,256)
    g_vector = tf.expand_dims(g,axis=1)
    score_vector = tf.reduce_sum(tf.multiply(L_vector,g_vector),2) #(batch,56*56)
    score = tf.nn.softmax(score_vector) #(batch,56*56)
    imgs = tf.expand_dims(score,axis=1)  #(batch,1,56*56)
    a_score = tf.expand_dims(score,axis=2) #(batch,56*56,1)
    gas = tf.multiply(L_vector, a_score)  # (batch,56*56,256)
    ga = tf.reduce_sum(gas,[1])
    print(ga.shape)
    print(imgs.shape)
    return ga,imgs,score

