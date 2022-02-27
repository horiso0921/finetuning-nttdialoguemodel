#! /bin/sh

# your own NAME (to be changed)
MODEL_NAME=$1
BASE_MODEL_NAME="base"
DATA_NAME=$MODEL_NAME
WORK_ROOT_DIR="/data/group1/z44384r/finetuning-nttdialoguemodel"

# your own path (to be changed)
WORK_DIR="$WORK_ROOT_DIR/model/${MODEL_NAME}"
DATA_DIR="$WORK_ROOT_DIR/data/PreprocessedBinaryData/$DATA_NAME"

# ここは事前学習済モデルのデータファイルパスを指定するところ
# 例えばNTTの事前学習済モデルなら以下のようにすること
if [ $2 = "BASE" ]; then
    PRETRAINED_MODEL="$WORK_ROOT_DIR/model/$BASE_MODEL_NAME/1.6B_2lhzhoam_4.92.pt"
    WORK_DIR=${WORK_DIR}_BASE
else
    if [ $2 = "PER" ]; then
        WORK_DIR=${WORK_DIR}_PER
        PRETRAINED_MODEL="$WORK_ROOT_DIR/model/$BASE_MODEL_NAME/persona50k-flat_1.6B_33avog1i_4.16.pt"
    else
        if [ $2 = "EMP" ]; then
            WORK_DIR=${WORK_DIR}_EMP
            PRETRAINED_MODEL="$WORK_ROOT_DIR/model/$BASE_MODEL_NAME/empdial50k-flat_1.6B_19jce27w_3.86.pt"
        else
            WORK_DIR=${WORK_DIR}_EMP_PER
            PRETRAINED_MODEL="$WORK_ROOT_DIR/model/$BASE_MODEL_NAME/emp_per.pt"
        fi
    fi
fi

# PRETRAINED_MODEL="/data/group1/z44384r/finetuning-nttdialoguemodel/model/$2/fine_tuned_models/checkpoint_best.pt"
# WORK_DIR=${WORK_DIR}_$2

WORK_DIR=${WORK_DIR}_${3}_${4}_${5}

# path & extension
MODEL_DIR="${WORK_DIR}/fine_tuned_models"
TENSORBOARD_DIR="${WORK_DIR}/fine_tuned_tensorboard_log"
SRC_LANG="src"
TRG_LANG="dst"

# model parameters
ENC_EMB=1920
ENC_FFN=7680
ENC_LAYER=2
ENC_HEAD=32
DEC_EMB=${ENC_EMB}
DEC_FFN=${ENC_FFN}
DEC_LAYER=24
DEC_HEAD=${ENC_HEAD}

# optimizer setting
GPU=0,1,2,3
# WARMUP_STEP=100  # {100, 500, 1000, 5000}
INIT_LR=1e-04
# LR=1e-04  # {1e-04, 5e-05, 1e-05, 5e-06}
MIN_LR=1e-09
WARMUP_STEP=$3
BATCH_SIZE=$4
LR=$5

# save & log
KEEP_LAST_EPOCH=1
KEEP_LAST_UPD=5
LOG_UPD=20

# else
SEED=1
