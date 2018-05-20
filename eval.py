#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KWS test module."""
import os

import numpy as np
import tensorflow as tf

import utils
from model import KWS

# Parameters.

# Data loading parameters.
tf.app.flags.DEFINE_string(
    "positive_data_file", "./data/pos_test.npy",
    "Data source for positive data.")
tf.app.flags.DEFINE_string(
    "negative_data_file", "./data/neg_test.npy",
    "Data source for negative data.")

# Evaluate parameters.
tf.app.flags.DEFINE_integer(
    "batch_size", 32,
    "Batch size (default: 64).")
tf.app.flags.DEFINE_string(
    "checkpoint_dir", "./models/7271/checkpoints",
    "Checkpoint directory from training run.")

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


def evaluate():
    # Load data.
    x_test, y = utils.load_data_and_labels(FLAGS.positive_data_file,
                                           FLAGS.negative_data_file)
    y_test = np.argmax(y, axis=1)

    # Evaluation.
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables.
            saver = tf.train.import_meta_graph(
                "{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholder from the graph by name.
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            cnn_dropout_keep_prob = graph.get_operation_by_name(
                "cnn_dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate.
            predictions = graph.get_operation_by_name(
                "softmax/predictions").outputs[0]

            # Generate batches for one epoch.
            batches = utils.batch_iter(
                list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions.
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(
                    predictions,
                    {input_x: x_test_batch,
                     cnn_dropout_keep_prob: 1.0})
                all_predictions = np.concatenate(
                    [all_predictions, batch_predictions])

    # Calculate accuracy.
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:.4}".format(correct_predictions/float(len(y_test))))

    # Calculate precision and recall.
    if y_test is not None:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and all_predictions[i] == 1.0:
                true_positive += 1
            if y_test[i] == 0.0 and all_predictions[i] == 1.0:
                false_positive += 1
            if y_test[i] == 1.0 and all_predictions[i] == 0.0:
                false_negative += 1
            if y_test[i] == 0.0 and all_predictions[i] == 0.0:
                true_negative += 1
        precision = 100.0 * true_positive / (true_positive + false_positive)
        recall = 100.0 * true_positive / (true_positive + false_negative)
        print("Precision: {:.4}".format(precision))
        print("Recall: {:.4}".format(recall))
        print("true_positive: {}".format(true_positive))
        print("false_positive: {}".format(false_positive))
        print("false_negative: {}".format(false_negative))
        print("true_negative: {}".format(true_negative))


def main(_):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
