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

r"""Finetune models for multi-backone inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import head2toe.models.finetune as finetune_models

from sklearn import ensemble as skens
from sklearn import feature_selection as skfs
import tensorflow.compat.v2 as tf


class FinetuneFS(finetune_models.Finetune):
  """A `tf.keras.Model` implementation of Finetune Feature Selection (FS).

  This learner trains a linear classifier on top of the features given.
  Additionally it supports having a processing step where a subset of features
  are selected.
  """

  def _select_fraction(self, scores, keep_fraction, keep_fraction_offset,
                       mean_interpolation_coef=0):
    """Given a scoring function returns the indices of high scores features."""
    n_kept = keep_fraction_offset + int(tf.size(scores).numpy() * keep_fraction)
    if mean_interpolation_coef > 0:
      # We need to interpolate the scores towards it's mean.
      scores, _ = self._interpolate_scores_towards_mean(
          scores, mean_interpolation_coef)
    _, sorted_indices = tf.nn.top_k(scores, k=n_kept)
    selected_indices = sorted_indices[keep_fraction_offset:]
    return selected_indices

  def _interpolate_scores_towards_mean(self, scores, coef):
    new_scores = []
    mean_scores = []
    for c_scores in tf.split(scores, self.embedding_sizes):
      c_score_mean = tf.reduce_mean(c_scores)
      mean_scores.append(c_score_mean)
      c_scores = c_scores * (1 - coef) + c_score_mean * coef
      new_scores.append(c_scores)
    return tf.concat(new_scores, 0), mean_scores

  def _broadcast_indices(self, kept_indices_all):
    """Splits and removes the offset for indices."""
    start_index = 0
    selected_feature_indices = []
    for embedding_size in self.embedding_sizes:
      end_index = start_index + embedding_size
      kept_indices = tf.boolean_mask(
          kept_indices_all,
          tf.math.logical_and(
              kept_indices_all >= start_index, kept_indices_all < end_index))
      # Remove the offset.
      kept_indices -= start_index

      start_index = end_index
      selected_feature_indices.append(kept_indices)
    return selected_feature_indices

  def _calculate_scores(self, learning_config, dataset):
    # Pre-generate the embeddings
    config_fs = learning_config.feature_selection
    with tf.device('/CPU:0'):
      embeddings, labels = self._embed_dataset(dataset)
    if config_fs.type == 'random':
      concat_embeddings = tf.concat(embeddings, -1)
      all_scores = tf.random.normal(concat_embeddings.shape[1:])
    elif config_fs.type.startswith('variance'):
      concat_embeddings = tf.concat(embeddings, -1)
      all_scores = tf.math.reduce_std(concat_embeddings, axis=0)
    elif config_fs.type.startswith('sklearn'):
      concat_embeddings = tf.concat(embeddings, -1)
      f_name = '_'.join(config_fs.type.strip().split('_')[1:])
      if f_name == 'trees':
        all_scores = skens.ExtraTreesClassifier(n_estimators=50).fit(
            concat_embeddings, labels).feature_importances_
      else:
        score_fn = getattr(skfs, f_name)
        all_scores = skfs.SelectPercentile(score_fn).fit(
            concat_embeddings, labels).scores_
    elif config_fs.type.startswith('connectivity'):
      # We don't care about query performance yet.
      new_config = copy.deepcopy(learning_config)
      # We don't do future selection in this innerloop.
      new_config.feature_selection.type = 'none'
      if new_config.feature_selection.is_overwrite:
        overwrite_dict = new_config.feature_selection.learning_config_overwrite
        for k, v in overwrite_dict.items():
          setattr(new_config, k, v)
      output_head = self._optimize_finetune(new_config, dataset, None,
                                            return_output_head=True)
      # TODO
      weights = output_head.layers[0].kernel

      if config_fs.type == 'connectivity_l1':
        all_scores = tf.reduce_sum(tf.abs(weights), axis=1)
      elif config_fs.type == 'connectivity_l2layer':
        if learning_config.group_lrp_is_embedding:
          # Here we match the groups used in embedding-based
          # group-regularization.
          new_scores = []
          for group in tf.split(weights, self.embedding_sizes, axis=0):
            score = tf.norm(group, axis=1, ord=2)
            group_norm = tf.norm(tf.reshape(group, [-1]), ord=2)
            new_scores.append(tf.ones_like(score) * group_norm)
          all_scores = tf.concat(new_scores, 0)
        else:
          # This is regular feature wise calculation, followed by averaging.
          all_scores = tf.norm(weights, axis=1, ord=2)
          # Score of each feature is equal to the layer score.
          new_scores = [
              tf.ones_like(score) * tf.reduce_mean(score) for score
              in tf.split(all_scores, self.embedding_sizes)]
          all_scores = tf.concat(new_scores, 0)
      elif config_fs.type == 'connectivity_l2':
        all_scores = tf.reduce_sum(weights**2, axis=1)
      elif config_fs.type == 'connectivity_linf':
        all_scores = tf.reduce_max(tf.abs(weights), axis=1)
      else:
        raise ValueError(f'config_fs.type: {config_fs.type} not valid')
      del output_head, weights
    else:
      raise ValueError(f'{config_fs.type} is not valid')
    return all_scores

  def _select_features(self, dataset, learning_config):
    config_fs = learning_config.feature_selection
    if config_fs.type == 'none':
      return None, None
    if config_fs.average_over_k > 1:
      all_scores = []
      for _ in range(config_fs.average_over_k):
        all_scores.append(self._calculate_scores(learning_config, dataset))
      all_scores = tf.reduce_mean(tf.stack(all_scores), 0)
    else:
      all_scores = self._calculate_scores(learning_config, dataset)
    kept_indices_all = self._select_fraction(
        all_scores, config_fs.keep_fraction, config_fs.keep_fraction_offset,
        mean_interpolation_coef=config_fs.mean_interpolation_coef)
    _, mean_scores = self._interpolate_scores_towards_mean(all_scores, 1.)
    selected_feature_indices = self._broadcast_indices(kept_indices_all)
    return selected_feature_indices, mean_scores

  def evaluate(self, learning_config, support_dataset, query_dataset,
               fs_dataset=None):
    """Performs evaluation on an episode.

    Args:
      learning_config: a `ConfigDict` specifying the learning configuration.
      support_dataset: a `tf.data.Dataset` for the support set.
      query_dataset: a `tf.data.Dataset` for the query set.
      fs_dataset: dataset or None. If given, features are
        selected using the datasets given.

    Returns:
      metrics: dict mapping metric names to metrics.
    """
    # Maybe select features.
    if fs_dataset:
      selected_feature_indices, mean_scores = self._select_features(
          fs_dataset, learning_config)
    else:
      selected_feature_indices, mean_scores = self._select_features(
          support_dataset, learning_config)
    metrics = self._optimize_finetune(
        learning_config, support_dataset, query_dataset,
        selected_feature_indices=selected_feature_indices)
    return_dict = self._process_metrics(metrics)

    if selected_feature_indices:
      for backbone_name, sfi, mean_score in zip(
          self.backbone_names, selected_feature_indices, mean_scores):
        return_dict[f'fs_count_{backbone_name}/'] = tf.size(sfi).numpy()
        return_dict[f'fs_meanscore_{backbone_name}/'] = mean_score.numpy()
        return_dict[f'pickle_sfi_{backbone_name}'] = sfi.numpy()
    return return_dict
