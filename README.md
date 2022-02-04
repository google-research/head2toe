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

python head2toe/evaluate.py --config=head2toe/configs_eval/finetune.py:imagenetr50 \
--config.dataset='data.caltech101'
```

## Disclaimer
This is not an officially supported Google product.
