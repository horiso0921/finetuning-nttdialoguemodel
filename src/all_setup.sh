conda create -n train-nn python=3.8.10
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate train-nn
cd ~/fairseq 
pip install --editable ./
pip install sentencepiece 
pip install tensorboard
