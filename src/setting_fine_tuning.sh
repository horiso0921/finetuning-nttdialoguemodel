#! /bin/sh

# your own NAME (to be changed)
MODEL_NAME="mymodel"
BASE_MODEL_NAME="BASE"
DATA_NAME="sample"

# your own path (to be changed)
WORK_DIR="../model/$MODEL_NAME"
DATA_DIR="../data/PreprocessedBinaryData/$DATA_NAME"

# ここは事前学習済モデルのデータファイルパスを指定するところ
# 例えばNTTの事前学習済モデルなら以下のようにすること
PRETRAINED_MODEL="../model/$BASE_MODEL_NAME/1.6B_2lhzhoam_4.92.pt"

# path & extension
MODEL_DIR="${WORK_DIR}/fine_tuned_models"
TENSORBOARD_DIR="${WORK_DIR}/fine_tuned_tensorboard_log"
SRC_LANG="context"
TRG_LANG="response"

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
MAX_UPDATE=`expr 10000 + 400000`  # fine-tuning updates + pre-training updates
UFREQ=1
WARMUP_STEP=5000  # {100, 500, 1000, 5000}
INIT_LR=1e-07
LR=1e-04  # {1e-04, 5e-05, 1e-05, 5e-06}
MIN_LR=1e-09

# save & log
KEEP_LAST_EPOCH=1
KEEP_LAST_UPD=5
LOG_UPD=20

# else
SEED=2020
