"""Contains logic for constructing the graph of the model."""

import tensorflow as tf


BASE_PAIRS = ['A', 'T', 'G', 'C']


class Model(object):
  """A model for predicting protein binding.

  This model requires tables to be initialized via
  tf.tables_initializer() before it is run by a session.
  """
  def __init__(self,
               sequence_length,
               mode,
               learning_rate,
               momentum_rate):
    """Constructs a model. This constructor creates its graph.

    Args:
      sequence_length: The length of a sequence.
      mode: The initial mode (tf.contrib.learn.ModeKeys.TRAIN, etc).
      learning_rate: The learning rate.
      momentum_rate: The momentum rate of the optimizer.
    """
    self.sequence_length = sequence_length
    self.mode = mode
    self.learning_rate = learning_rate
    self.momentum_rate = momentum_rate
    self.sequences_placeholder = tf.placeholder(tf.string)
    self.true_labels_placeholder = tf.placeholder(tf.int32)

    activation_function = tf.sigmoid

    encoded = self._OneHotEncodeSequences(self.sequences_placeholder)

    # Perform a convolution and then max pool.
    conv_layer = tf.contrib.layers.convolution2d(
        inputs=encoded,
        num_outputs=4,  # The number of filters to use.
        kernel_size=(1, 14),
        activation_fn=activation_function,
        normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={
            "is_training": self.mode == tf.contrib.learn.ModeKeys.TRAIN
        },
        padding='VALID')
    max_pool = tf.nn.max_pool(
        conv_layer,
        ksize=[1, 1, 6, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')asdfasd
    flattened_activations = tf.contrib.layers.flatten(max_pool)

    # Add a couple dense layers.
    fully_connected_layer = tf.contrib.layers.fully_connected(
        flattened_activations,
        num_outputs=64,
        activation_fn=activation_function)
    fully_connected_layer = tf.contrib.layers.fully_connected(
        fully_connected_layer,
        num_outputs=32,
        activation_fn=activation_function)
    fully_connected_layer = tf.contrib.layers.fully_connected(
        fully_connected_layer,
        num_outputs=16,
        activation_fn=activation_function)

    self.logits_layer = tf.contrib.layers.fully_connected(
        fully_connected_layer,
        num_outputs=2,
        activation_fn=activation_function)
    one_hot_targets = tf.contrib.layers.one_hot_encoding(
        labels=self.true_labels_placeholder, num_classes=2)

    # The shape from the one hot encoding is for some reason
    # (BATCH SIZE, 1, 2). We squeeze to remove the dimension of 1.
    self.one_hot_targets = tf.squeeze(one_hot_targets)
    class_predictions = tf.cast(
        tf.argmax(self.logits_layer, 1), tf.int32)

    self.loss_op = tf.losses.softmax_cross_entropy(
        self.one_hot_targets, self.logits_layer)
    self.accuracy_op = tf.contrib.metrics.accuracy(
        class_predictions, self.true_labels_placeholder)
    self.train_op = tf.contrib.layers.optimize_loss(
        self.loss_op,
        tf.contrib.framework.get_global_step(),
        optimizer=tf.train.MomentumOptimizer(self.learning_rate,
                                             self.momentum_rate),
        learning_rate=self.learning_rate)


  def _OneHotEncodeSequences(self, sequences):
    """One-hot encodes string DNA sequences.

    Args:
      sequences: A tensor of multiple string DNA sequences.

    Returns:
      A tensor with sequences one-hot encoded as matrices.
    """
    # One-hot encode the DNA sequences.
    split_sequence = tf.string_split(sequences, delimiter='')
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(BASE_PAIRS + ['N']), num_oov_buckets=0)
    indices = table.lookup(split_sequence.values)
    embeddings = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                              [0, 0, 0, 1], [0, 0, 0, 0]])
    encoded = tf.to_float(tf.nn.embedding_lookup(embeddings, indices))
    return tf.reshape(
        encoded, (-1, 1, self.sequence_length, len(BASE_PAIRS)))
