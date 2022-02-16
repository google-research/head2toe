# Head2Toe: Utilizing Intermediate Representations for Better OOD Generalization
Code for reproducing our results in the Head2Toe paper.
<img src="https://github.com/google-research/head2toe/blob/main/imgs/h2t.png" alt="Head2Toe " width="80%" align="middle">

**Paper**: [arxiv.org/abs/2201.03529](https://arxiv.org/abs/2201.03529)

## Setup
First clone this repo.
```bash
git clone https://github.com/google-research/head2toe.git
cd head2toe
```

We need to download the pre-trained ImageNet checkpoints. If you use the code
below it will move the checkpoints under the correct folder. If you use a
different name you need to update paths in `head2toe/configs_eval/finetune.py`.
```bash
mkdir checkpoints
cd checkpoints
wget -c https://storage.googleapis.com/gresearch/head2toe/imagenetr50.tar.gz
wget -c https://storage.googleapis.com/gresearch/head2toe/imagenetvitB16.tar.gz
tar -xvf imagenetr50.tar.gz
tar -xvf imagenetvitB16.tar.gz
rm *.tar.gz
cd ../
```

Let's run some tests. The following script creates a virtual environment and
installs the necessary libraries. Finally, it runs a few tests.
```bash
bash run.sh
```

We need to activate the virtual environment before running an experiment. With
that, we are ready to run some trivial Caltech101 experiments.
```bash
source env/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD

python head2toe/evaluate.py \
--config=head2toe/configs_eval/finetune.py:imagenetr50 \
--config.eval_mode='test' --config.dataset='data.caltech101'
```

Note that running evaluation for each task requires downloading and
preparing multiple datasets, which can take up-to a day. Please check out
https://github.com/google-research/task_adaptation for more details on
installing the datasets.

## Running Head2Toe
Our results presented in Table-1 of our paper can be reproduced by running the
following command for Caltech-101 task. This takes 15-10mins on a single V100
gpu.
```bash
python head2toe/evaluate.py \
--config=head2toe/configs_eval/finetune_h2t.py:imagenetr50 \
--config.dataset='data.caltech101' \
--config.eval_mode='test' --config.learning.cached_eval=False \
--config.backbone.additional_features_target_size=8192 \
--config.learning.feature_selection.keep_fraction=0.01 \
--config.learning.feature_selection.learning_config_overwrite.group_lrp_regularizer_coef=0.00001 \
--config.learning.learning_rate=0.01 --config.learning.training_steps=5000 \
--config.learning.log_freq=1000
```
Hyper-parameters used for different tasks can be found in the appendix. Here is
the command for dSprites-Orientation task.
```bash
python head2toe/evaluate.py \
--config=head2toe/configs_eval/finetune_h2t.py:imagenetr50 \
--config.dataset='data.dsprites(predicted_attribute="label_orientation",num_classes=16)' \
--config.eval_mode='test' --config.learning.cached_eval=False \
--config.backbone.additional_features_target_size=512 \
--config.learning.feature_selection.keep_fraction=0.2 \
--config.learning.feature_selection.learning_config_overwrite.group_lrp_regularizer_coef=0.00001 \
--config.learning.learning_rate=0.01 --config.learning.training_steps=500 \
--config.learning.log_freq=1000
```

## Running other baselines.
- **Regularization Baselines**: Use `finetune_h2t.py` config together with
`l1_regularizer`, `l2_regularizer` or `group_lrp_regularizer_coef` flags.
- **Linear**: Use `finetune.py` config.

Set `config.learning.finetune_backbones` to true for enabling the finetuning of
the backbone for any experiment. If you like to run any other experiments or
if you have questions, feel free to create a new issue.

## Disclaimer
This is not an officially supported Google product.
