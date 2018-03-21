"""Reads datasets and provides them."""

import csv


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
    for row in reader:
      if not row:
        continue
      print `row`
