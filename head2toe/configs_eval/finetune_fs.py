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

"""Default configutation."""
from head2toe.configs_eval import finetune
from ml_collections import ConfigDict


def get_config(config_string):
  config = finetune.get_config(config_string)
  config['model_name'] = 'FinetuneFS'
  new_learning_config = ConfigDict({
      'feature_selection':
          ConfigDict({
              # Following types exist: 'connectivity_mask',
              # 'connectivity_l1', 'random', 'none', 'variance' and
              # 'sklearn_x' where x in
              # [chi2, f_classif, mutual_info_classif, trees]
              'type': 'none',
              'fs_dataset': '',
              'is_overwrite': False,
              'average_over_k': 1,
              'keep_fraction': 0.1,
              'keep_fraction_offset': 0,
              'mean_interpolation_coef': 0.,
              'learning_config_overwrite':
                  ConfigDict({
                      'group_lrp_regularizer_coef': 1e-4,
                      'finetune_backbones': False,
                  })
          }),
  })
  config['learning'].update(new_learning_config)
  print(f'Config backbone: {config.backbone}')
  return config
