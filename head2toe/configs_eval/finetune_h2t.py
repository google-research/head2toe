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
from head2toe.configs_eval import finetune_fs


def get_config(config_string):
  config = finetune_fs.get_config(config_string)
  if 'imagenetr50' in config_string:
    all_blocks = set({'after_root', 'logits', 'pre_logits_pre_pooling'})
    for i, j in enumerate([3, 4, 6, 3]):
      for k in range(j):
        for l in range(3):
          all_blocks.add(f'block{i+1}_unit{k+1}_layer{l}')
  elif 'imagenetvitB16' in config_string:
    all_blocks = set({'cls_embedded', 'encoded_sequence',
                      'position_embedded_input', 'root_output_with_cls'})
    for i in range(12):
      all_blocks.add(f'encoder_{i}_attn')
      all_blocks.add(f'encoder_{i}_mlp_1')
      all_blocks.add(f'encoder_{i}_mlp_2')
      all_blocks.add(f'encoder_{i}_pre_attn')
  else:
    raise ValueError(f'This config is not supported for {config_string}')
  config.backbone.additional_features = ','.join(all_blocks)
  config.learning.feature_selection.type = 'connectivity_l2'
  config.learning.feature_selection.is_overwrite = True
  print(f'Config backbone: {config.backbone}')
  return config
