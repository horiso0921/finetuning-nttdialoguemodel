DATA_NAME=$1
PRE_DATA_DIR=/data/group1/z44384r/finetuning-nttdialoguemodel/data/PreprocessedData/$DATA_NAME
AFTER_DATA_DIR=/data/group1/z44384r/finetuning-nttdialoguemodel/data/PreprocessedBinaryData/$DATA_NAME
TRAIN=$PRE_DATA_DIR/train
DEV=$PRE_DATA_DIR/valid
PRE_PROCESSED_DIR=$AFTER_DATA_DIR
SPM_VOCAB=/data/group1/z44384r/finetuning-nttdialoguemodel/model/spm/sp_oall_32k.txt
rm -r $AFTER_DATA_DIR
SRC_LANG="src"
TRG_LANG="dst"
N_WORKER=12

echo "Train-context:" ${TRAIN}.${SRC_LANG}
echo "Train-response:" ${TRAIN}.${TRG_LANG}
echo "Dev-context:" ${DEV}.${SRC_LANG}
echo "Dev-response:" ${DEV}.${TRG_LANG}
echo "Output:" ${PRE_PROCESSED_DIR}
echo
echo "sentencepiece vocab:" ${SPM_VOCAB}
echo


echo "Create:" ${FAIRSEQ_VOCAB}
echo
echo "Your fairseq version:"
pip list | grep fairseq
echo

set -x

fairseq-preprocess \
  --source-lang ${SRC_LANG} \
  --target-lang ${TRG_LANG} \
  --trainpref ${TRAIN} \
  --validpref ${DEV} \
  --destdir ${PRE_PROCESSED_DIR} \
  --srcdict ${SPM_VOCAB} \
  --joined-dictionary \
  --workers ${N_WORKER}