#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -S
#PJM -o log/finetune.log

module load cuda/10.2.89_440.33.01 openmpi_cuda/4.0.4 nccl/2.7.3
eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate train-nn
nvidia-smi
SRC_NAMES="sample"
WARMUP_STEPS=500
LR=1e-04
BATCH_SIZE=16

dir=${SRC_NAMES}_${WARMUP_STEPS}_${BATCH_SIZE}_${LR}
mkdir -p ${dir}
bash ../src/preprocess.sh $SRC_NAMES > ${dir}/preprocess_`date "+%Y%m%d-%H%M"`.log 2> ${dir}/preprocess_error_`date "+%Y%m%d-%H%M"`.log
bash ../src/train_fine_tuning.sh $SRC_NAMES "BASE" $WARMUP_STEPS $BATCH_SIZE $LR > ${dir}/train_`date "+%Y%m%d-%H%M"`.log 2> ${dir}/train_error_`date "+%Y%m%d-%H%M"`.log

bash ../src/eval_test.sh ${dir} > ${dir}/test_`date "+%Y%m%d-%H%M"`.log 2> ${dir}/test_error_`date "+%Y%m%d-%H%M"`.log
