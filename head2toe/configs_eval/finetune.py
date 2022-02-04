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

# Lint as: python3
"""Default configutation."""
import re
from ml_collections import ConfigDict


def get_config(config_string):
  train_batch_size = 128
  eval_batch_size = 50
  config = ConfigDict({
      'dataset':
          'data.caltech101',
      'eval_mode':
          'valid',
      'is_vtab_5fold_valid':
          True,
      'seed':
          8,
      'max_num_gpus':
          1,
      'learning':
          ConfigDict({
              'optimizer': 'adam',  #  adadelta, adadelta_adaptive, sgd
              'learning_rate': 0.1,
              'grad_clip_value': -1.,  # Applied if positive.
              'l1_regularizer': 0.,
              'l2_regularizer': 0.,
              'group_lrp_regularizer_coef': 0.,
              'group_lrp_regularizer_r': 2.,
              'group_lrp_regularizer_p': 1.,
              'group_lrp_is_embedding': False,
              'training_steps': 500,
              'data_fraction': 1.,
              'cached_eval': True,
              'use_cosine_decay': True,
              'train_batch_size': train_batch_size,
              'eval_batch_size': eval_batch_size,
              'finetune_backbones': False,
              'finetune_lr_multiplier': 1.,
              'finetune_steps_multiplier': 1.,
              # ('', 'unit_vector', 'per_feature')
              'feature_normalization': 'unit_vector',
              # nohidden, random_100, random_1000, trainable_100, trainable_1000
              'output_head_type': 'nohidden',
              'output_head_zeroinit': False,
              'log_freq': 50,
          }),
      'model_name':
          'Finetune'
  })

  config.backbone = get_backbone_config(config_string)
  print(f'Config backbone: {config.backbone}')
  return config


def get_backbone_config(config_string):
  """Gets backbone configuration according to the key given."""
  # Example patterns:
  # imagenetr50, imagenetr50_2x
  pattern = r'^([A-Za-z0-9]+)?_?(\d+)?x?'
  searched = re.search(pattern, config_string)
  if not searched:
    raise ValueError(f'Unrecognized config_string: {config_string}')
  added_backbone, n_repeat = searched.groups()
  print(f'Split config: {added_backbone}, {n_repeat}')
  processed_names = []
  processed_handles = []
  processed_signatures = []
  processed_output_keys = []
  input_sizes = tuple()

  if added_backbone in SINGLE_MODELS:
    n_repeat = int(n_repeat) if n_repeat else 1
    processed_names += [added_backbone] * n_repeat
    handle, size = SINGLE_MODELS[added_backbone]
    if isinstance(handle, list):
      processed_handles = handle * n_repeat
      processed_handles = processed_handles[:n_repeat]
    else:
      processed_handles += [handle] * n_repeat

    if 'vit' in added_backbone:
      processed_signatures += ['serving_default'] * n_repeat
      processed_output_keys += ['pre_logits'] * n_repeat
    else:
      processed_signatures += ['representation'] * n_repeat
      processed_output_keys += ['pre_logits'] * n_repeat
    input_sizes += (size,) * n_repeat
  else:
    raise ValueError(f'added_backbone:{added_backbone} is not recognized')

  return ConfigDict({
      'names': processed_names,
      'handles': processed_handles,
      'signatures': processed_signatures,
      'output_keys': processed_output_keys,
      'input_sizes': input_sizes,
      'include_input': False,
      'additional_features': '',
      'additional_features_pool_size': 0,
      'cls_token_pool': 'normal',
      # If target size is provided, pool size is ignored.
      'additional_features_target_size': 0,
      'additional_features_multi_target_sizes': '',
  })


SINGLE_MODELS = {
    'imagenetr50': ('checkpoints/imagenetr50/', 240),
    'imagenetvitB16': ('checkpoints/imagenetvitB16/', 224)
}
