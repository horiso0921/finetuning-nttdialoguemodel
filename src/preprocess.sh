python ../src/_prepro_spm.py
DATA_NAME=$1
PRE_DATA_DIR=../data/PreprocessedData/$DATA_NAME
AFTER_DATA_DIR=../data/PreprocessedBinaryData/$DATA_NAME
TRAIN=$PRE_DATA_DIR/train
DEV=$PRE_DATA_DIR/valid
TEST=$PRE_DATA_DIR/test
PRE_PROCESSED_DIR=$AFTER_DATA_DIR
SPM_VOCAB=../model/spm/sp_oall_32k.txt
rm -r $AFTER_DATA_DIR
SRC_LANG="src"
TRG_LANG="dst"
N_WORKER=12

echo "Train-context:" ${TRAIN}.${SRC_LANG}
echo "Train-response:" ${TRAIN}.${TRG_LANG}
echo "Dev-context:" ${DEV}.${SRC_LANG}
echo "Dev-response:" ${DEV}.${TRG_LANG}
echo "Test-context:" ${TEST}.${SRC_LANG}
echo "Test-response:" ${TEST}.${TRG_LANG}
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
  --testpref ${TEST} \
  --destdir ${PRE_PROCESSED_DIR} \
  --srcdict ${SPM_VOCAB} \
  --joined-dictionary \
  --workers ${N_WORKER}