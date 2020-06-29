# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="./data"
#MODELROOT="${HOME}/deepcluster_models"
#MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"
MODEL
EXP="./deepcluster_exp/linear_classif"

PYTHON="${HOME}/test/conda/bin/python"

mkdir -p ${EXP}


${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 3 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12


  #python eval_linear.py --model ./checkpoint.pth.tar --data ./data --conv 3 --lr 0.01 --wd -7 --tencrops --verbose --exp ./deepcluster_exp --workers 12
  #python eval_linear.py --model ./alexnet/checkpoint.pth.tar --data ./data --conv 3 --lr 0.01 --wd -7 --tencrops --verbose --exp ./deepcluster_alexnet_pretrained_exp --workers 4