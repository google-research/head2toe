# coding=utf-8
# Copyright 2021 Head2Toe Authors.
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

"""Trivial tests for checking models evaluates episodes without an error."""
from head2toe.configs_eval import finetune
from head2toe.configs_eval import finetune_fs
import head2toe.models.finetune as finetune_models
import head2toe.models.finetune_fs as finetune_fs_models
import tensorflow.compat.v2 as tf


class FinetuneTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.config = finetune.get_config('JFTr50_solo')
    self.sur_model = finetune_models.Finetune(self.config)

  def test_evaluate(self):
    """Tests whether the model runs with no-error using dummy inputs."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([4, 240, 240, 3]),
         tf.random.uniform([4,], maxval=2, dtype=tf.int32))).batch(2)
    results = self.sur_model.evaluate(self.config.learning, dataset, dataset,
                                      False)
    print(results)


class FinetuneFSTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.config = finetune_fs.get_config('JFTr50_solo')
    self.sur_model = finetune_fs_models.FinetuneFS(self.config)

  def test_evaluate(self):
    """Tests whether the model runs with no-error using dummy inputs."""
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.uniform([4, 240, 240, 3]),
         tf.random.uniform([4,], maxval=2, dtype=tf.int32))).batch(2)
    results = self.sur_model.evaluate(self.config.learning, dataset, dataset,
                                      False)
    print(results)

if __name__ == '__main__':
  tf.test.main()
