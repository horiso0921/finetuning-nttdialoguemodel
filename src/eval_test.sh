DATA_NAME=sample
DATA_NAME=../data/PreprocessedBinaryData/${DATA_NAME}

fairseq-validate $DATA_NAME \
--path ../model/$1/fine_tuned_models/checkpoint_best.pt \
--task translation \
--source-lang src \
--target-lang dst \
--batch-size 2 \
--ddp-backend no_c10d \
--valid-subset test \
--skip-invalid-size-inputs-valid-test