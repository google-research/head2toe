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
import math
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
# nohidden: just output layer.
# random_100: 1 non-trainable hidden layer with 100 units.
# random_1000: ... with 1000 units.
# trainable_100: 1 trainable hidden layer with 100 units.
# trainable_1000: ... with 1000 units.
OUTPUT_HEAD_TYPES = ['nohidden', 'random', 'trainable']


def zero_aware_normalize(embedding, axis):
  """If the norm is zero leaves the row unnormalized."""
  # Following will have nans when the norm of vector(the divider) is zero.
  normalized, norms = tf.linalg.normalize(embedding, axis=axis)
  is_zero_norm = tf.broadcast_to(tf.equal(norms, 0), normalized.shape)
  return tf.where(is_zero_norm, tf.zeros_like(embedding), normalized)


def _check_and_convert(norm_ord):
  """Validates the order is positive or 'inf'."""
  if isinstance(norm_ord, (float, int)) and norm_ord > 0:
    return norm_ord
  elif isinstance(norm_ord, str) and norm_ord == 'inf':
    return np.inf
  else:
    raise ValueError(f'norm_order:{norm_ord} is not valid')


class GroupLRP(tf.keras.regularizers.Regularizer):
  """A regularizer that applies Group L-r/p penalty to the weights.

  The L-r/p regularization penalty is computed as:
  `loss = coef * norm(norm(x, ord=r, axis=1), ord=p)`

  Attributes:
      coef: Float; regularization factor.
      r: int, Must be >0. or 'inf'
      p: int, Must be >0. or 'inf'
      group_sizes: iterable or None; used to split feature vector into tensors.
  """

  def __init__(self, coef=0., r=2, p=1, group_sizes=None):
    self.coef = tf.keras.backend.cast_to_floatx(coef)
    self.r = _check_and_convert(r)
    self.p = _check_and_convert(p)
    self.group_sizes = group_sizes

  def __call__(self, x):
    regularization = tf.keras.backend.constant(0., dtype=x.dtype)
    if self.coef:
      if self.group_sizes:
        group_norms = []
        for group in tf.split(x, self.group_sizes, axis=0):
          group_norms.append(tf.norm(tf.reshape(group, [-1]), ord=self.r))
        regularization += tf.norm(tf.stack(group_norms), ord=self.p)
      else:
        regularization += self.coef * tf.norm(
            tf.norm(x, axis=1, ord=self.r), ord=self.p)
    return regularization

  def get_config(self):
    return {'coef': float(self.coef), 'r': self.r, 'p': self.p,
            'group_sizes': tuple(self.group_sizes)}


class Finetune(tf.keras.Model):
  """A `tf.keras.Model` implementation of Finetune.

  This learner trains a linear classifier on top of the features given.
  TODO Split finetune_backbone implementation from linear probe.
  """

  def __init__(self, config):
    """Initializes a `Finetune` instance.

    Args:
      config: a `ConfigDict` specifying the backbones configuration.
    """
    super(Finetune, self).__init__()
    self._backbone_config = config.backbone
    self._learning_config = config.learning
    available_gpus = tf.config.list_physical_devices(device_type='GPU')
    if config.max_num_gpus > len(available_gpus):
      logging.warning('config.max_num_gpus: %s > n_gpus', config.max_num_gpus)
    else:
      available_gpus = available_gpus[:config.max_num_gpus]
    logging.info('N_GPUS: %d in use', len(available_gpus))
    # To get /physical_device:GPU:0
    available_gpus = [':'.join(g.name.split(':')[1:]) for g in available_gpus]
    self.strategy = tf.distribute.MirroredStrategy(devices=available_gpus)
    with self.strategy.scope():
      res = self.load_backbones()
      self.backbones, self.backbone_names, self.embedding_sizes = res

  def load_backbones(self):
    backbone_config = self._backbone_config
    # Load pre-trained backbones.
    backbones = []
    backbone_names = []
    embedding_sizes = []
    for name, handle, signature, output_key, size in zip(
        backbone_config.names,
        backbone_config.handles,
        backbone_config.signatures,
        backbone_config.output_keys,
        backbone_config.input_sizes):

      if self._learning_config.finetune_backbones:
        backbone = hub.KerasLayer(handle, trainable=True)
      elif signature is None:
        backbone = hub.KerasLayer(handle, trainable=False)
      else:
        backbone = hub.KerasLayer(
            handle, signature=signature, output_key=None,
            trainable=False, signature_outputs_as_dict=True)
      inputs = tf.keras.Input(shape=(None, None, 3))
      resized_inputs = inputs
      if size is not None:
        inputs = tf.keras.Input(shape=(size, size, 3))
        resized_inputs = tf.image.resize(inputs, size=[size, size])
      outputs = backbone(resized_inputs)
      if backbone_config.additional_features:
        updated_outputs = []
        all_output_keys = [output_key]
        if backbone_config.include_input:
          outputs['input'] = resized_inputs
          all_output_keys.append('input')
        all_output_keys.extend(
            backbone_config.additional_features.strip().split(','))
        if backbone_config.additional_features_multi_target_sizes:
          t_sizes = backbone_config.additional_features_multi_target_sizes
          target_embedding_sizes = t_sizes.strip().split(',')
        else:
          target_embedding_sizes = [
              backbone_config.additional_features_target_size]
        # TODO Probably use the function to get multiple pooled features.
        # It should also return the names maybe.
        # Also it might be more straight forward to use pool_sizes.
        for target_embedding_size in target_embedding_sizes:
          new_outputs = flatten_and_concat(
              outputs, output_keys=all_output_keys,
              pool_size=backbone_config.additional_features_pool_size,
              target_size=int(target_embedding_size),
              cls_token_pool=backbone_config.cls_token_pool)
          new_names = [f'{name}_{n}_{target_embedding_size}' for n
                       in all_output_keys]
          for newname, out in zip(new_names, new_outputs):
            logging.info('Backbone name: %s, shape: %s', newname, out.shape)
            backbone_names.append(newname)
            embedding_sizes.append(out.shape[-1])
          updated_outputs += new_outputs
        outputs = updated_outputs
      else:
        outputs = outputs[output_key]
        logging.info('Backbone name: %s, shape: %s', name, outputs.shape)
        backbone_names.append(name)
        embedding_sizes.append(outputs.shape[-1])
      backbone = tf.keras.Model(inputs=inputs, outputs=outputs)
      backbones.append(backbone)
    return backbones, backbone_names, embedding_sizes

  def _get_optimizer(self, learning_config, n_classes):
    learning_rate = learning_config.learning_rate
    clipvalue = (learning_config.grad_clip_value
                 if learning_config.grad_clip_value > 0 else None)
    if learning_config.use_cosine_decay:
      learning_rate = tf.keras.experimental.CosineDecay(
          learning_rate, learning_config.training_steps)
    optimizer = learning_config.optimizer
    if optimizer == 'adam':
      optimizer = tf.optimizers.Adam(learning_rate, clipvalue=clipvalue)
    elif optimizer == 'sgd':
      optimizer = tf.optimizers.SGD(learning_rate, momentum=0.9,
                                    clipvalue=clipvalue)
    else:
      raise ValueError('Unknown optimizer')
    return optimizer

  def _embed_batch(self, x, is_training=False):
    """Compute the feature representation of a batch.

    Args:
      x: input tensor.
      is_training: bool, passed to the backbone.
    Returns:
      embedding_list: A list of tf.Tensors.
    """
    embedding_list = []
    for backbone in self.backbones:
      output_backbone = backbone(x, training=is_training)
      # Note that the output of the backbone can be a list.
      if isinstance(output_backbone, list):
        for out in output_backbone:
          embedding_list.append(out)
      else:
        embedding_list.append(output_backbone)
    return embedding_list

  def _embed_dataset(self, dataset):
    """Compute the feature representation of a batch.

    Args:
      dataset: a `tf.data.Dataset` corresponding to the support or query set.
    Returns:
      embeddings: A list of tf.Tensors.
      labels: A tf.Tensor.
    """
    batch_embedding_lists = []
    labels = []
    for x, y in dataset:
      labels.append(y)
      batch_embedding_lists.append(self._embed_batch(x))

    labels = tf.concat(labels, axis=0)
    output_embeddings = []
    for i in range(len(batch_embedding_lists[0])):
      embedding_i = [batch[i] for batch in batch_embedding_lists]
      output_embeddings.append(tf.concat(embedding_i, axis=0))
    return output_embeddings, labels

  def _process_metrics(self, metrics):
    (support_loss_iter, support_accuracy_iter, query_loss_iter,
     query_accuracy_iter) = metrics

    ret_dict = {
        'support_loss': support_loss_iter[-1],
        'support_accuracy': support_accuracy_iter[-1],
        'query_loss': query_loss_iter[-1],
        'query_accuracy': query_accuracy_iter[-1]
                }

    return ret_dict

  def _process_embeddings(self, embeddings, selected_features,
                          normalization='unit_vector'):
    """Processes embeddings by normalizing an concatenating.

    Args:
      embeddings: list of Tensors, where each Tensor is the embeddings
        of a particular backbone.
      selected_features: list of Tensors, where each Tensor indicates the
        indices to be selected.
      normalization: str, 'unit_vector', 'per_feature_std'.
        'unit_vector' SUR style normalization
        'per_feature' similar to Batch-Normalization

    Returns:
      flattened and possibly scaled embeddings.
    """
    # shape= (n_image, n_features)
    assert normalization in ('unit_vector', 'per_feature', '')
    if selected_features:
      # Following removes the backbones altogether if no feature is selected.
      embeddings = [
          tf.gather(embedding, indices, axis=1) for embedding, indices
          in zip(embeddings, selected_features)
          if np.prod(indices.shape) > 0
      ]
    if normalization == 'unit_vector':
      embeddings = [zero_aware_normalize(e, axis=1) for e in embeddings]
    embeddings = tf.concat(embeddings, -1)
    if normalization == 'per_feature':
      # Normalize each feature to have unit variance and zero mean.
      mean, var = tf.nn.moments(embeddings, axes=0)
      bn_args = {'offset': None,
                 'scale': None,
                 'variance_epsilon': 1e-5}
      embeddings = tf.nn.batch_normalization(
          embeddings, mean, var, **bn_args)
    return embeddings

  @tf.function(reduce_retracing=True)
  def _compute_loss_and_accuracy(self, output_head, logits, labels,
                                 global_batch_size=None):
    """Computes the loss and accuracy on an episode."""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    if output_head.losses:
      loss += tf.add_n(output_head.losses)
    accuracy = tf.cast(tf.equal(
        tf.math.argmax(logits, axis=1, output_type=labels.dtype),
        labels), tf.float32)
    loss = tf.nn.compute_average_loss(loss, global_batch_size=global_batch_size)
    accuracy = tf.nn.compute_average_loss(accuracy,
                                          global_batch_size=global_batch_size)
    return loss, accuracy

  def _init_training_vars(self, num_ways, learning_config):
    if learning_config.l1_regularizer or learning_config.l2_regularizer:
      regularizer = tf.keras.regularizers.L1L2(
          l1=learning_config.l1_regularizer, l2=learning_config.l2_regularizer)
    elif learning_config.group_lrp_regularizer_coef:
      group_sizes = (self.embedding_sizes
                     if learning_config.group_lrp_is_embedding else None)
      regularizer = GroupLRP(
          coef=learning_config.group_lrp_regularizer_coef,
          group_sizes=group_sizes,
          r=learning_config.group_lrp_regularizer_r,
          p=learning_config.group_lrp_regularizer_p)
    else:
      regularizer = None
    output_layers = []
    assert any(learning_config.output_head_type.startswith(t)
               for t in OUTPUT_HEAD_TYPES)
    if learning_config.output_head_type.startswith('random'):
      n_units = int(learning_config.output_head_type.split('_')[1])
      n_units = n_units if n_units > 0 else num_ways
      output_layers.append((n_units, False, 'relu'))
    elif learning_config.output_head_type.startswith('trainable'):
      n_units = int(learning_config.output_head_type.split('_')[1])
      n_units = n_units if n_units > 0 else num_ways
      output_layers.append((n_units, True, 'relu'))
    # Final layer, this is the only layer when type=nohidden.
    output_layers.append((num_ways, True, None))
    output_head = tf.keras.Sequential()
    for i, (n_features, is_trainable, f_activation) in enumerate(output_layers):
      kwargs_dense = {'trainable': is_trainable}
      if i == 0 and learning_config.output_head_zeroinit:
        logging.info('First layer in output head is zero initialized.')
        kwargs_dense['kernel_initializer'] = tf.zeros
      new_layer = tf.keras.layers.Dense(
          n_features, activation=f_activation, kernel_regularizer=regularizer,
          **kwargs_dense)
      output_head.add(new_layer)
    return output_head

  def evaluate(self, learning_config, support_dataset, query_dataset,
               **unused_kwargs):
    """Performs evaluation on an episode.

    Args:
      learning_config: a `ConfigDict` specifying the learning configuration.
      support_dataset: a `tf.data.Dataset` for the support set.
      query_dataset: a `tf.data.Dataset` for the query set.

    Returns:
      metrics: dict mapping metric names to metrics.
    """
    metrics = self._optimize_finetune(
        learning_config, support_dataset, query_dataset)
    return_dict = self._process_metrics(metrics)
    return return_dict

  def _optimize_finetune(self, learning_config, support_dataset, query_dataset,
                         selected_feature_indices=None,
                         return_output_head=False):
    """Optimize the output layers and possibly the backbones, too.

    Args:
      learning_config: A ConfigDict.
      support_dataset: a `tf.data.Dataset` for the support set.
      query_dataset: a `tf.data.Dataset` or None for the query set.
      selected_feature_indices: defines which features are selected.
      return_output_head: bool that decides what is returned. If true, this
        means we return the trained output head and also don't calculate query
        performance.
    Returns:
      lambda_logits: The optimized lambdas or None.
      support_loss_iter: A list of iterate values.
      support_accuracy_iter: A list of iterate values.
      query_loss_iter: A list of iterate values.
      query_accuracy_iter: A list of iterate values.
    """
    if selected_feature_indices:
      # Print statistics
      for name, indices in zip(self.backbone_names, selected_feature_indices):
        logging.info('Backbone: %s, selected %d', name, len(indices))
    # Pre-generate the embeddings
    with tf.device('/CPU:0'):
      support_embeddings, support_labels = self._embed_dataset(support_dataset)
      # Normalize the data
      support_representation = self._process_embeddings(
          support_embeddings, selected_feature_indices,
          normalization=learning_config.feature_normalization)
      logging.info('Support representation shape: %s',
                   support_representation.shape)
      support_representation_ph = support_representation[:1]
    if query_dataset is None:
      query_labels = tf.constant([], dtype=support_labels.dtype)
    elif learning_config.cached_eval:
      with tf.device('/CPU:0'):
        query_embeddings, query_labels = self._embed_dataset(query_dataset)
        query_representation = self._process_embeddings(
            query_embeddings, selected_feature_indices,
            normalization=learning_config.feature_normalization)
    else:
      # We still need query labels to get number of classes. In some situations
      # Support set might not have all classes, and then our output head
      # would be smaller, which creates NaNs in loss calculation when
      # tf.nn.sparse_softmax_cross_entropy_with_logits is used.
      all_labels = []
      for _, batch_labels in query_dataset:
        all_labels.append(batch_labels)
      query_labels = tf.concat(all_labels, axis=0)
    support_loss_iter = []
    support_accuracy_iter = []
    query_loss_iter = []
    query_accuracy_iter = []
    all_labels = tf.concat([support_labels, query_labels], 0)
    num_ways = tf.cast(tf.math.reduce_max(tf.unique(all_labels)[0]) + 1,
                       tf.int32)
    with self.strategy.scope():
      if (learning_config.finetune_backbones and
          learning_config.finetune_lr_multiplier != 1.):
        learning_config = copy.deepcopy(learning_config)
        new_lr = (learning_config['learning_rate'] *
                  learning_config.finetune_lr_multiplier)
        learning_config['learning_rate'] = new_lr
        logging.info('Finetuning learning rate is updated to %s',
                     new_lr)
      if (learning_config.finetune_backbones and
          learning_config.finetune_steps_multiplier != 1.):
        learning_config = copy.deepcopy(learning_config)
        new_steps = int(learning_config['training_steps'] *
                        learning_config.finetune_steps_multiplier)
        learning_config['training_steps'] = new_steps
        logging.info('Finetuning training steps are updated to %s', new_steps)
      optimizer = self._get_optimizer(learning_config, num_ways)
      output_head = self._init_training_vars(num_ways, learning_config)
      # Initialize the layer
      output_head(support_representation_ph)

    training_vars = output_head.trainable_variables
    if learning_config.finetune_backbones:
      for b in self.backbones:
        training_vars.extend(b.trainable_variables)
    else:
      batch_size = learning_config.train_batch_size
      # Regenerate the dataset with precomputed embeddings.
      support_dataset = tf.data.Dataset.from_tensor_slices(
          (support_representation, support_labels)).batch(batch_size)
      if query_dataset and learning_config.cached_eval:
        query_dataset = tf.data.Dataset.from_tensor_slices(
            (query_representation, query_labels)).batch(
                learning_config.eval_batch_size)
    logging.info('Trainable variables: %s', [v.name for v in training_vars])
    support_dataset = support_dataset.repeat()
    dist_support_dataset = self.strategy.experimental_distribute_dataset(
        support_dataset)
    if query_dataset:
      dist_query_dataset = self.strategy.experimental_distribute_dataset(
          query_dataset)

    @tf.function()
    def _train_step(x, y):
      with tf.GradientTape() as tape:
        if learning_config.finetune_backbones:
          # Pass the images through multiple backbones.
          embeddings = self._embed_batch(x, is_training=True)
          # Normalize the data
          x = self._process_embeddings(
              embeddings, selected_feature_indices,
              normalization=learning_config.feature_normalization)
        logits = output_head(x)
        loss, accuracy = self._compute_loss_and_accuracy(
            output_head, logits, y,
            global_batch_size=learning_config.train_batch_size)
      grads = tape.gradient(loss, training_vars)
      optimizer.apply_gradients(zip(grads, training_vars))
      return loss, accuracy

    @tf.function()
    def _eval_step(x, y):
      if (learning_config.finetune_backbones or
          not learning_config.cached_eval):
        # Pass the images through multiple backbones.
        embeddings = self._embed_batch(x, is_training=False)
        # Normalize the data
        x = self._process_embeddings(
            embeddings, selected_feature_indices,
            normalization=learning_config.feature_normalization)
      logits = output_head(x)
      loss, accuracy = self._compute_loss_and_accuracy(
          output_head, logits, y,
          global_batch_size=learning_config.eval_batch_size)
      return loss, accuracy

    for i, (x, y) in enumerate(dist_support_dataset):
      if i == learning_config.training_steps:
        break
      per_replica_results = self.strategy.run(_train_step, args=(x, y))
      pr_loss, pr_acc = per_replica_results
      support_loss = self.strategy.reduce(
          tf.distribute.ReduceOp.SUM, pr_loss, axis=None)
      support_accuracy = self.strategy.reduce(
          tf.distribute.ReduceOp.SUM, pr_acc, axis=None)
      if query_dataset and (i % learning_config.log_freq == 0 or
                            i == (learning_config.training_steps - 1)):
        logging.info('Evaluating at iteration: %d', i)
        all_losses = []
        all_accs = []
        for query_x, query_y in dist_query_dataset:
          per_replica_results = self.strategy.run(_eval_step, args=(
              query_x, query_y))
          pr_loss, pr_acc = per_replica_results
          c_query_loss = self.strategy.reduce(
              tf.distribute.ReduceOp.SUM, pr_loss, axis=None)
          c_query_accuracy = self.strategy.reduce(
              tf.distribute.ReduceOp.SUM, pr_acc, axis=None)
          all_losses.append(c_query_loss)
          all_accs.append(c_query_accuracy)
        # This assumes all batches are same size, so ensure that.
        query_loss_iter.append(np.mean(all_losses))
        query_accuracy_iter.append(np.mean(all_accs))
      support_loss_iter.append(support_loss.numpy())
      support_accuracy_iter.append(support_accuracy.numpy())
    if learning_config.finetune_backbones:
      # Reload the backbones to reset any changes made during finetuning.
      with self.strategy.scope():
        self.backbones, _, _ = self.load_backbones()
    if return_output_head:
      return output_head
    else:
      del output_head.layers[0].kernel
      return (support_loss_iter, support_accuracy_iter, query_loss_iter,
              query_accuracy_iter)


def flatten_and_concat(output_dict, output_keys, pool_size=0, target_size=0,
                       cls_token_pool='normal'):
  """Summarizes a dict of outputs into single feature vector."""
  # If target_size is given pool_size is ignored.
  if cls_token_pool not in ('normal', 'only_cls', 'nopool_cls'):
    raise ValueError("%s must be one of 'normal', 'only_cls', 'nopool_cls'"
                     % cls_token_pool)
  all_features = []
  for k in output_keys:
    output = output_dict[k]
    # TODO Make this more readable by making each branch a function.
    if len(output.shape) == 4:
      if target_size > 0:
        # Overwrite pool size so that final output matches target_size as close
        # as possible.
        _, width, _, channels = output.shape
        if channels >= target_size:
          # Global pool.
          pool_size = 0
        else:
          # Assuming square image.
          n_patches_per_row = int(math.sqrt(target_size // channels))
          pool_size = width // n_patches_per_row
      if pool_size > 0:
        output = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size, strides=pool_size)(output)
        all_features.append(tf.keras.layers.Flatten()(output))
      else:
        # Global pool
        all_features.append(tf.reduce_mean(output, axis=[1, 2]))
    elif len(output.shape) == 3:
      if cls_token_pool == 'only_cls':
        output = output[:, 0, :]
      else:
        if cls_token_pool == 'nopool_cls':
          # We will get the cls as it is and pool the rest.
          cls_output, output = output[:, 0, :], output[:, 1:, :]
        if target_size > 0:
          # Overwrite pool size so that final output matches target_size as
          # close as possible.
          _, n_token, channels = output.shape
          if channels >= target_size:
            # Global pool.
            pool_size = 0
          else:
            # Assuming square image.
            n_groups = target_size / channels
            pool_size = int(n_token / n_groups)
        if pool_size > 0:
          output = tf.keras.layers.AveragePooling1D(
              pool_size=pool_size, strides=pool_size)(output)
          output = tf.keras.layers.Flatten()(output)
        else:
          # Global pool
          output = tf.reduce_mean(output, axis=[1])
        if cls_token_pool == 'nopool_cls':
          output = tf.concat([cls_output, output], axis=1)
      all_features.append(output)
    elif len(output.shape) == 2:
      all_features.append(output)
    else:
      raise ValueError(
          f'Output tensor: {k} with shape {output.shape} not 2D or 4D.')
  return all_features
