"""Reads datasets and provides them."""

import csv
import numpy as np


class Dataset(object):

  def __init__(self, data_file_path):
    """Constructs a dataset.

    A dataset may pertain to for instance train or eval.

    Args:
      data_file_path: A path to a .data file. These test files have a
        DNA sequence separated by a space from either 0 or 1 depending
        on the true label for that row.

    Raises:
      ValueError: If the data being read is not well-formed. For
        instance, all sequences must have the same length.
    """
    reader = csv.reader(open(data_file_path), delimiter=' ')
    sequences = []
    labels = []
    self._sequence_length = None

    for row in reader:
      if not row:
        # Some files may contain empty rows. Filter them out.
        continue

      sequence = row[1].upper()
      if self._sequence_length is None:
        # Determine the sequence length from the data.
        self._sequence_length = len(sequence)
      elif self._sequence_length != len(sequence):
        # All sequences must have the same length.
        raise ValueError(
            'Sequence %r has length %d, not %d' % (
                sequence, len(sequence), self._sequence_length))

      sequences.append(sequence)
      labels.append(int(row[2]))

    self._sequences = np.asarray(sequences)
    self._labels = np.asarray(labels)

  def GetCount(self):
    """Gets the count of examples.

    Returns:
      The number of examples in the dataset.
    """
    return len(self._sequences)

  def GetSequenceLength(self):
    """Gets the length of a sequence in the dataset.

    Returns:
      The length of a sequence.
    """
    return self._sequence_length

  def GetBatch(self, size):
    """Gets a random batch of `size` examples.

    Args:
      size: The size of the batch to obtain. Must be less than the count
        of examples.

    Returns:
      A 2-tuple of numpy arrays. The first one contains sampled
      examples. The second one contains the labels. The rows correspond
      to each other.
    """
    indices = np.random.choice(self.GetCount(), size, replace=False)
    return (self._sequences[indices], self._labels[indices])
