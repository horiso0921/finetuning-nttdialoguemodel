#! /bin/sh

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

date
hostname
uname -a
which python
python --version
pip list

. setting_fine_tuning.sh

mkdir -p ${MODEL_DIR}

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train ${DATA_DIR} \
  --finetune-from-model ${PRETRAINED_MODEL} \
  --save-dir ${MODEL_DIR} \
  --seed ${SEED} \
  --source-lang ${SRC_LANG} \
  --target-lang ${TRG_LANG} \
  --arch transformer \
  --dropout 0.1 \
  --attention-dropout 0.0 \
  --relu-dropout 0.0 \
  --encoder-embed-dim ${ENC_EMB} \
  --encoder-ffn-embed-dim ${ENC_FFN} \
  --encoder-layers ${ENC_LAYER} \
  --encoder-attention-heads ${ENC_HEAD} \
  --encoder-normalize-before \
  --decoder-embed-dim ${DEC_EMB} \
  --decoder-ffn-embed-dim ${DEC_FFN} \
  --decoder-layers ${DEC_LAYER} \
  --decoder-attention-heads ${DEC_HEAD} \
  --decoder-normalize-before \
  --fp16 \
  --memory-efficient-fp16 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-updates ${WARMUP_STEP} \
  --warmup-init-lr ${INIT_LR} \
  --lr ${LR} \
  --stop-min-lr ${MIN_LR} \
  --weight-decay 0.0 \
  --clip-norm 0.1 \
  --patience 5 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --log-format simple \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --keep-last-epochs ${KEEP_LAST_EPOCH} \
  --log-interval ${LOG_UPD} \
  --batch-size 16 \
  > ${MODEL_DIR}/train.log
