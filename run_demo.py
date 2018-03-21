"""This demo trains + evals a model predicting protein binding. """

import dataset

if __name__ == '__main__':
  train_dataset = dataset.Dataset(
      'data/wgEncodeAwgTfbsSydhHelas3Stat3IggrabUniPk/train.data')
