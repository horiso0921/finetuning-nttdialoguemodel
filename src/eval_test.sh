DATA_NAME=$1
DATA_NAME=../data/PreprocessedBinaryData/${DATA_NAME}

fairseq-validate $DATA_NAME \
--path ../model/base/1.6B_2lhzhoam_4.92.pt \
--task translation \
--source-lang src \
--target-lang dst \
--batch-size 2 \
--ddp-backend no_c10d \
--valid-subset test \
--skip-invalid-size-inputs-valid-test