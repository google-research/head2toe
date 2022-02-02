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

"""Tests for input_pipeline."""
import absl.testing.parameterized as parameterized
from head2toe import input_pipeline
import tensorflow as tf


class InputPipelineTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (84, 2, 'train', 'test'),
      (84, 2, 'eval', 'test'),
      (84, 1000, 'train', 'test'),
      (84, 1000, 'eval', 'test'),
      (240, 2, 'train', 'test'),
      (240, 2, 'train', 'valid'),
      (240, 2, 'eval', 'valid'),
  )
  def test_vtab_pipeline(self, image_size, batch_size, mode, eval_mode):
    data_source = 'data.caltech101'
    dataset = input_pipeline.create_vtab_dataset(
        dataset=data_source, mode=mode, image_size=image_size,
        batch_size=batch_size, eval_mode=eval_mode)
    if batch_size <= 1000:
      x, y = next(iter(dataset))
      self.assertAllEqual(x.shape, [batch_size, image_size, image_size, 3])
      self.assertAllEqual(y.shape, [batch_size])
    if batch_size == 1000 and mode == 'train':
      # Full batch.
      self.assertLen(list(iter(dataset)), 1)

if __name__ == '__main__':
  tf.test.main()
