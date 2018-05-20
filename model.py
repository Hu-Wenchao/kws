#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWS models"""
import math

import tensorflow as tf


class KWS(object):
    """
    Attention-based End-to-End Models for Small-Footprint Keyword Spotting.
    Uses an encoder layer, followed by an attention layer, 
    linear transformation and softmax layer.
    """

    def __init__(self, num_steps, num_freqs, num_classes,
                 rnn_type, rnn_layers, num_units, att_mechanism,
                 att_size, use_cnn, num_channel):

        # Placeholders for input, output, dropout, and l2_loss.
        self.input_x = tf.placeholder(
            tf.float32, [None, num_steps, num_freqs], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.l2_reg_lambda = tf.placeholder(tf.float32, name="l2_reg_lambda")
        self.cnn_dropout_keep_prob = tf.placeholder(
            tf.float32, name="cnn_dropout_keep_prob")

        # Keeping track of l2 regularization loss.
        l2_loss = tf.constant(0.0)

        # CNN layer.
        # input shape: [batch_size, num_steps, num_freqs]
        # output shape:
        with tf.device("/gpu:0"), tf.name_scope("cnn"):
            if use_cnn:
                # shape: [batch_size, num_steps, num_freqs, 1]
                cnn_input = tf.expand_dims(self.input_x, -1)
                filter_shape = [20, 5, 1, num_channel]
                cnn_W = tf.Variable(
                    tf.random_normal(filter_shape), name="cnn_W")
                cnn_b = tf.Variable(
                    tf.constant(0.0, shape=[num_channel]), name="cnn_b")
                conv = tf.nn.conv2d(
                    cnn_input,
                    cnn_W,
                    strides=[1, 1, 2, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity.
                cnn_res = tf.nn.relu(tf.nn.bias_add(conv, cnn_b), name="relu")
                # Add dropout.
                cnn_dropout = tf.nn.dropout(
                    cnn_res, self.cnn_dropout_keep_prob)
                # Flatten.
                output_height = math.floor((num_steps - 20 + 1) / 1)
                output_width = math.floor((num_freqs - 5 + 1) / 2)
                self.cnn_output = tf.reshape(
                    cnn_dropout,
                    [-1, output_height, output_width * num_channel])
            else:
                self.cnn_output = self.input_x

        # Encoder layer.
        # input shape: [batch_size, num_steps, num_freqs] -> no cnn
        #              [batch_size, output_height, ow * nc] -> with cnn
        # output shape: [batch_size, num_steps, num_units] -> no cnn
        #               [batch_size, output_height, num_units] -> with cnn
        with tf.name_scope("encoder_layer"):
            if rnn_type == "rnn":
                cells = [tf.nn.rnn_cell.BasicRNNCell(num_units)
                         for _ in range(rnn_layers)]
            elif rnn_type == "lstm":
                cells = [tf.nn.rnn_cell.LSTMCell(num_units)
                         for _ in range(rnn_layers)]
            else:
                cells = [tf.nn.rnn_cell.GRUCell(num_units)
                         for _ in range(rnn_layers)]
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            batch_size = tf.shape(self.cnn_output)[0]
            initial_state = multi_rnn_cell.zero_state(batch_size, tf.float32)
            self.encoder_output, state = tf.nn.dynamic_rnn(
                multi_rnn_cell,
                self.cnn_output,
                initial_state=initial_state,
                dtype=tf.float32)

        # Average attention.
        # input shape: [batch_size, num_steps, num_units]
        # output shape: [batch_size, num_units]
        with tf.name_scope("attention"):
            if att_mechanism == "average":
                self.c_output = tf.reduce_mean(self.encoder_output, 1)
            else:
                # Soft attention.
                # shape: [num_units, att_size]
                att_W = tf.Variable(tf.random_normal(
                    [num_units, att_size], stddev=0.1))
                # shape: [att_size]
                att_b = tf.Variable(tf.random_normal(
                    [att_size], stddev=0.1))
                # shape: [att_size]
                att_v = tf.Variable(tf.random_normal(
                    [att_size], stddev=0.1))
                # shape: [batch_size, num_steps, att_size]
                att_t = tf.tanh(tf.tensordot(
                    self.encoder_output, att_W, axes=1) + att_b)
                # shape: [batch_size, num_steps]
                att_et = tf.tensordot(att_t, att_v, axes=1, name="att_et")
                # shape: [batch_size, num_steps]
                att_alpha = tf.nn.softmax(att_et, name="att_alpha")
                # shape: [batch_size, num_units]
                self.c_output = tf.reduce_sum(
                    self.encoder_output * tf.expand_dims(att_alpha, -1), 1)

        # Softmax and output.
        # input shape: [batch_size, num_units]
        # output shape: [batch_size, num_classes] -> self.logits
        #               [batch_size] -> self.predictions
        with tf.name_scope("softmax"):
            softmax_W = tf.Variable(
                tf.random_normal([num_units, num_classes]),
                name="softmax_W")
            softmax_b = tf.Variable(
                initial_value=tf.constant(0.0, shape=[num_classes]),
                name="softmax_b")
            l2_loss += tf.nn.l2_loss(softmax_W)
            l2_loss += tf.nn.l2_loss(softmax_b)
            self.logits = tf.nn.xw_plus_b(
                self.c_output, softmax_W, softmax_b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss.
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + \
                self.l2_reg_lambda * l2_loss

        # Accuracy.
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
