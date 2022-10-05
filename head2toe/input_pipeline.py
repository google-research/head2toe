# coding=utf-8
# Copyright 2022 Head2Toe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline functions for VTAB and Meta-Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging

import numpy as np
from task_adaptation import data_loader
import tensorflow.compat.v2 as tf


def _filter_to_k_shot(dataset, num_classes, k):
  """Filters k-shot subset from a dataset."""
  # !!! IMPORTANT: the dataset should *not* be shuffled. !!!
  # Make sure that `shuffle_buffer_size=1` in the call to
  # `dloader.get_tf_data`.

  # Indices of included examples in the k-shot balanced dataset.
  keep_example = []
  # Keep track of the number of examples per class included in
  # `keep_example`.
  class_counts = np.zeros([num_classes], dtype=np.int32)
  for _, label in dataset.as_numpy_iterator():
    # If there are less than `k` examples of class `label` in `example_indices`,
    # keep this example and update the class counts.
    keep = class_counts[label] < k
    keep_example.append(keep)
    if keep:
      class_counts[label] += 1
    # When there are `k` examples for each class included in `keep_example`,
    # stop searching.
    if (class_counts == k).all():
      break

  dataset = tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(keep_example),
      dataset
  )).filter(lambda keep, _: keep).map(lambda _, example: example).cache()

  return dataset


def create_vtab_dataset_balanced(dataset, image_size, batch_size,
                                 data_fraction):
  """Creates a VTAB input_fn to be used by `tf.Estimator`.

  Deterministic balanced sampling from vtab datasets.

  Args:
    dataset: str, VTAB task to evaluate on.
    image_size: int
    batch_size: int
    data_fraction: float, used to calculate n_shots

  Returns:
    input_fn, input function to be passed to `tf.Estimator`.
  """
  dloader = data_loader.get_dataset_instance(
      {'dataset': dataset, 'data_dir': None})
  num_classes = dloader.get_num_classes()
  n_shots = max(int(1000 * data_fraction / num_classes), 1)
  logging.info('n_shots: %d', n_shots)
  def _dict_to_tuple(batch):
    return batch['image'], batch['label']
  dataset = dloader.get_tf_data(
      split_name='trainval',
      batch_size=batch_size,
      preprocess_fn=functools.partial(
          data_loader.preprocess_fn,
          input_range=(-1.0, 1.0),
          size=image_size),
      epochs=0,
      drop_remainder=False,
      for_eval=False,
      shuffle_buffer_size=1,
      prefetch=1,
      train_examples=None,
  ).unbatch().map(_dict_to_tuple)
  filtered_dataset = _filter_to_k_shot(dataset, num_classes, n_shots)
  return filtered_dataset.shuffle(1000).batch(batch_size)


def create_vtab_dataset(dataset, image_size, batch_size, mode,
                        eval_mode='test', valid_fold_id=4):
  """Creates a VTAB input_fn to be used by `tf.Estimator`.

  Note: There is one episode/VTAB dataset.

  Args:
    dataset: str, VTAB task to evaluate on.
    image_size: int
    batch_size: int
    mode: str in {'train', 'eval'}, whether to build the input function for
      training or evaluation.
    eval_mode: str in {'valid', 'test'}, whether to build the input functions
      for validation or test runs.
    valid_fold_id: int, 0 <= valid_fold_id < 5, valid_fold_id=4 corresponds to
      the default value in VTAB.

  Returns:
    input_fn, input function to be passed to `tf.Estimator`.
  """
  assert 0 <= valid_fold_id < 5
  dloader = data_loader.get_dataset_instance(
      {'dataset': dataset, 'data_dir': None})
  if mode not in ('train', 'eval'):
    raise ValueError("mode should be 'train' or 'eval'")
  is_training = mode == 'train'

  def _dict_to_tuple(batch):
    return batch['image'], batch['label']
  if eval_mode == 'test':
    split_name = 'train800val200' if is_training else 'test'
  elif eval_mode == 'valid':
    val_start, val_end = valid_fold_id * 200, (valid_fold_id + 1) * 200
    if is_training:
      split_name = f'train[:{val_start}]+train[{val_end}:1000]'
    else:
      split_name = f'train[{val_start}:{val_end}]'
    logging.info('Using split_name: %s', split_name)

    if split_name not in dloader._tfds_splits:
      dloader._tfds_splits[split_name] = split_name
      dloader._num_samples_splits[split_name] = 800 if is_training else 200
  else:
    raise ValueError(f'eval_mode: {eval_mode} invalid')

  return dloader.get_tf_data(
      split_name=split_name,
      batch_size=batch_size,
      preprocess_fn=functools.partial(
          data_loader.preprocess_fn,
          input_range=(-1.0, 1.0),
          size=image_size),
      epochs=0,
      drop_remainder=False,
      for_eval=not is_training,
      # Our training data has at most 1000 samples, therefore a shuffle buffer
      # size of 1000 is sufficient.
      shuffle_buffer_size=1000,
      prefetch=1,
      train_examples=None,
  ).map(_dict_to_tuple)
