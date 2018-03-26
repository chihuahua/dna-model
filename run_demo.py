"""This demo trains + evals a model predicting protein binding. """

import argparse
import os
import sys

import dataset
import model as model_lib
import tensorflow as tf


_DATA_DIRECTORY = 'data/wgEncodeAwgTfbsSydhHelas3Stat3IggrabUniPk'


def main(_):
  with tf.Session() as sess:
    train_dataset = dataset.Dataset(
        os.path.join(_DATA_DIRECTORY, 'train.data'))
    test_dataset = dataset.Dataset(
        os.path.join(_DATA_DIRECTORY, 'test.data'))

    model = model_lib.Model(
        sequence_length=train_dataset.GetSequenceLength(),
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        learning_rate=FLAGS.learning_rate,
        momentum_rate=FLAGS.momentum_rate)

    sess.run(
        [tf.global_variables_initializer(), tf.tables_initializer()])
    for i in range(FLAGS.max_steps):
      train_batch = train_dataset.GetBatch(FLAGS.train_batch_size)
      feed_dict = {
          model.sequences_placeholder: train_batch[0],
          model.true_labels_placeholder: train_batch[1],
      }
      model.mode = tf.contrib.learn.ModeKeys.TRAIN
      _ = sess.run([model.train_op, model.loss_op], feed_dict=feed_dict)

      model.mode = tf.contrib.learn.ModeKeys.EVAL
      loss = sess.run([model.loss_op], feed_dict=feed_dict)
      print `loss`


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--train_batch_size', type=int, default=100,
                      help='The batch size used for training.')
  parser.add_argument('--learning_rate', type=float, default=3e-1,
                      help='The learning rate.')
  parser.add_argument('--momentum_rate', type=float, default=0.1,
                      help='The momentum rate for the optimizer.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
