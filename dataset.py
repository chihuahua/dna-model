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
    """
    reader = csv.reader(open(data_file_path), delimiter=' ')
    sequences = []
    labels = []

    for row in reader:
      if not row:
        # Some files may contain empty rows.
        continue

      sequences.append(row[1])
      labels.append(int(row[2]))

    self._sequences = np.asarray(sequences)
    self._labels = np.asarray(labels)

  def GetCount(self):
    """Gets the count of examples.

    Returns:
      The number of examples in the dataset.
    """
    return len(self._sequences)

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
