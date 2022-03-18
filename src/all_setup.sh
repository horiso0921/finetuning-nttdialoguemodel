conda create -n train-nn python=3.8.10
conda activate train-nn
cd ~/fairseq 
pip install --editable ./
pip install sentencepiece 
pip install tensorboard