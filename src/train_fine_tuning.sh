#! /bin/sh

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

date
hostname
uname -a
which python
python --version
pip list


. ../src/setting_fine_tuning.sh $1 $2 $3 $4 $5

mkdir -p ${MODEL_DIR}

CUDA_VISIBLE_DEVICES=${GPU} fairseq-train ${DATA_DIR} \
  --restore-file ${PRETRAINED_MODEL} \
  --reset-dataloader \
  --reset-lr-scheduler \
  --reset-meters \
  --reset-optimizer \
  --ddp-backend no_c10d \
  --save-dir ${MODEL_DIR} \
  --seed ${SEED} \
  --source-lang ${SRC_LANG} \
  --target-lang ${TRG_LANG} \
  --arch transformer \
  --dropout 0.1 \
  --attention-dropout 0.1 \
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
  --share-decoder-input-output-embed \
  --memory-efficient-fp16 \
  --optimizer adafactor \
  --lr-scheduler inverse_sqrt \
  --warmup-updates ${WARMUP_STEP} \
  --lr ${LR} \
  --weight-decay 0.0 \
  --clip-norm 0.1 \
  --patience 10 \
  --update-freq 1 \
  --validate-interval-updates 100 \
  --min-loss-scale=1e-10 \
  --criterion cross_entropy \
  --log-format simple \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --log-interval ${LOG_UPD} \
  --no-epoch-checkpoints \
  --no-last-checkpoints \
  --batch-size ${BATCH_SIZE}
