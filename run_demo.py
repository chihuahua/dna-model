"""This demo trains + evals a model predicting protein binding. """

import os

import dataset


_DATA_DIRECTORY = 'data/wgEncodeAwgTfbsSydhHelas3Stat3IggrabUniPk'


if __name__ == '__main__':
  train_dataset = dataset.Dataset(
      os.path.join(_DATA_DIRECTORY, 'train.data'))
  test_dataset = dataset.Dataset(
      os.path.join(_DATA_DIRECTORY, 'test.data'))
