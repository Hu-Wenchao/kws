#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================
# Author: Hu Wenchao (huwcbill@hotmail.com)
# Date: 2018/05/19
# Description:
#   Realization of model described in arxiv:1803.10916v1,
#   Attention-based End-to-End Models for Small-Footprint
#   Keyword Spotting.
# ==========================================================
"""KWS train module."""
import time
import os
import datetime

import numpy as np
import tensorflow as tf

import utils
from model import KWS

# Parameters.

# Data loading parameters.
tf.app.flags.DEFINE_float(
    "val_percentage", 0.1,
    "Percentage of the training data to use for validation.")
tf.app.flags.DEFINE_string(
    "positive_data_file", "./data/pos_train.npy",
    "Data source for positive data.")
tf.app.flags.DEFINE_string(
    "negative_data_file", "./data/neg_train.npy",
    "Data source for negative data.")
tf.app.flags.DEFINE_string(
    "positive_sample_dir", "./data/positive/",
    "Directory for positive samples.")
tf.app.flags.DEFINE_string(
    "negative_sample_dir", "./data/negative/",
    "Directory for negative samples.")
tf.app.flags.DEFINE_string(
    "data_dir", "./data/",
    "Directory for feature files.")

# Model hyperparameters.
tf.app.flags.DEFINE_integer(
    "num_steps", 99,
    "Number of steps for each input wave file (default: 99).")
tf.app.flags.DEFINE_integer(
    "num_freqs", 40,
    "Number of mel filters used (default: 40).")
tf.app.flags.DEFINE_integer(
    "num_classes", 2,
    "Number of classes (default: 2).")
tf.app.flags.DEFINE_string(
    "rnn_type", "gru",
    "RNN structure to be used, can be rnn/lstm/gru (default: lstm).")
tf.app.flags.DEFINE_integer(
    "rnn_layers", 1,
    "Number of rnn layers in the structure (default: 3).")
tf.app.flags.DEFINE_integer(
    "num_units", 128,
    "Number of hidden units in rnn (default: 64).")
tf.app.flags.DEFINE_float(
    "l2_reg_lambda", 1e-5,
    "L2 regularization lambda value (default: 1e-5).")
tf.app.flags.DEFINE_string(
    "att_mechanism", "soft",
    "Attention mechanism, can be average/soft (default: soft).")
tf.app.flags.DEFINE_integer(
    "att_size", 100,
    "Attention size (default: 50).")
tf.app.flags.DEFINE_boolean(
    "use_cnn", False,
    "Whether to add a CNN layer before RNNs (default: False).")
tf.app.flags.DEFINE_integer(
    "num_channel", 16,
    "Number of channel for CNN (default: 16).")
tf.app.flags.DEFINE_float(
    "cnn_dropout_keep_prob", 0.5,
    "CNN dropout keep prob (default: 0.5).")

# Training parameters.
tf.app.flags.DEFINE_integer(
    "batch_size", 32,
    "Batch size (default: 64).")
tf.app.flags.DEFINE_integer(
    "num_epochs", 200,
    "Number of training epochs (default: 200).")
tf.app.flags.DEFINE_integer(
    "evaluate_every", 100,
    "Evaluate model on dev set after this many steps (default: 100).")
tf.app.flags.DEFINE_integer(
    "checkpoint_every", 100,
    "Save model after this many steps (default: 100).")
tf.app.flags.DEFINE_integer(
    "num_checkpoints", 5,
    "Number of checkpoints to store (default: 5).")
tf.app.flags.DEFINE_string(
    "model_dir", "./models",
    "Save model and summary into this directory.")

# Misc parameters.
tf.app.flags.DEFINE_boolean(
    "allow_soft_placement", True,
    "Allow device soft device placement.")
tf.app.flags.DEFINE_boolean(
    "log_device_placement", False,
    "Log placement of ops on devices.")

FLAGS = tf.app.flags.FLAGS

# Show parameters.
print("Parameters:")
print("========================================================")
for param, value in sorted(FLAGS.flag_values_dict().items()):
    print("{} = {}".format(param.upper(), value))
print("========================================================")


def preprocess():
    # Data preparation.

    # Load data.
    print("Loading data...")
    if not os.path.exists(FLAGS.positive_data_file) or \
       not os.path.exists(FLAGS.negative_data_file):
        utils.prepare_data(FLAGS.positive_sample_dir,
                           FLAGS.negative_sample_dir,
                           FLAGS.data_dir)
    x_data, y = utils.load_data_and_labels(FLAGS.positive_data_file,
                                           FLAGS.negative_data_file)

    # Randomly shuffle data.
    np.random.seed(23)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_data[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/validation set.
    val_data_index = -1 * int(FLAGS.val_percentage * float(len(y)))
    x_train, x_val = x_shuffled[:val_data_index], x_shuffled[val_data_index:]
    y_train, y_val = y_shuffled[:val_data_index], y_shuffled[val_data_index:]

    print("Total number of training data: {}".format(len(x_data)))
    print("Train/Validation split: {}/{}".format(len(y_train), len(y_val)))

    # Remove unnecessary variables.
    del x_data, y, x_shuffled, y_shuffled

    return x_train, y_train, x_val, y_val


def train(x_train, y_train, x_val, y_val):
    # Training.
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            kws = KWS(
                num_steps=FLAGS.num_steps,
                num_freqs=FLAGS.num_freqs,
                num_classes=FLAGS.num_classes,
                rnn_type=FLAGS.rnn_type,
                rnn_layers=FLAGS.rnn_layers,
                num_units=FLAGS.num_units,
                att_mechanism=FLAGS.att_mechanism,
                att_size=FLAGS.att_size,
                use_cnn=FLAGS.use_cnn,
                num_channel=FLAGS.num_channel)

            # Define training procedure.
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # Piecewise decay learning rate.
            boundaries = [1000, 2000]
            values = [1e-3, 5e-4, 1e-4]
            lr = tf.train.piecewise_constant(global_step, boundaries, values)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            grads_and_vars = optimizer.compute_gradients(kws.loss)
            # Gradient norm clipping to [-1, 1].
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                          for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(
                capped_gvs, global_step=global_step)

            # Keep track of gradient values and sparsity.
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name),
                        tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries.
            timestamp = str(int(time.time()))[-4:]
            out_dir = os.path.abspath(os.path.join(FLAGS.model_dir, timestamp))
            print("Writing to {}]\n".format(out_dir))

            # Summaries for loss and accuracy.
            loss_summary = tf.summary.scalar("loss", kws.loss)
            acc_summary = tf.summary.scalar("accuracy", kws.accuracy)

            # Train summaries.
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "train_summary")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Validation summaries.
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "val_summary")
            val_summary_writer = tf.summary.FileWriter(
                val_summary_dir, sess.graph)

            # Checkpoint directory.
            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables.
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """A single traning step."""
                feed_dict = {
                    kws.input_x: x_batch,
                    kws.input_y: y_batch,
                    kws.l2_reg_lambda: FLAGS.l2_reg_lambda,
                    kws.cnn_dropout_keep_prob: FLAGS.cnn_dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op,
                     kws.loss, kws.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()[-10:]
                print("{}: step {:>4}, loss {:10.4}, acc {:10.4}".format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def val_step(x_batch, y_batch, writer=None):
                """Evaluate model on validation set."""
                feed_dict = {
                    kws.input_x: x_batch,
                    kws.input_y: y_batch,
                    kws.l2_reg_lambda: FLAGS.l2_reg_lambda,
                    kws.cnn_dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run(
                    [global_step, val_summary_op, kws.loss, kws.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()[-10:]
                print("{}: step {:>4}, loss {:10.4}, acc {:10.4}".format(
                    time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches.
            batches = utils.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size,
                FLAGS.num_epochs)

            # Training.
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluate:")
                    val_step(x_val, y_val, writer=val_summary_writer)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix,
                                      global_step=current_step)
                    print("Saved model checkpoint to {}".format(path))


def main(_):
    x_train, y_train, x_val, y_val = preprocess()
    train(x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    tf.app.run()
