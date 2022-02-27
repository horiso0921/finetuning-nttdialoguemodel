import glob
import sentencepiece as spm
import re
import os

import sys


SPM = "/data/group1/z44384r/finetuning-nttdialoguemodel/model/spm/sp_oall_32k.model"
sp = spm.SentencePieceProcessor()
sp.Load(SPM)
DATANAME = sys.argv[1]

SRC_DIR = "/data/group1/z44384r/finetuning-nttdialoguemodel/data/RawData/"+DATANAME
DATA_DIR = "/data/group1/z44384r/finetuning-nttdialoguemodel/data/PreprocessedData/"+DATANAME

if not os.path.exists(SRC_DIR):
    os.makedirs(SRC_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for name in ['train.dst', 'valid.dst', 'train.src', 'valid.src']:
    with open(SRC_DIR+"/"+name) as f, open(DATA_DIR+"/"+name, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)
