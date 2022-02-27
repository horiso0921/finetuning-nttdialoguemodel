import glob
import sentencepiece as spm
import re
import os

import sys


SPM = "../model/spm/sp_oall_32k.model"
sp = spm.SentencePieceProcessor()
sp.Load(SPM)

DATANAMES = glob.glob("../data/RawData/*")

for DATANAME in DATANAMES:
    try:
        DATANAME = DATANAME[16:]
        SRC_DIR = "../data/RawData/"+DATANAME
        DATA_DIR = "../data/PreprocessedData/"+DATANAME

        if not os.path.exists(SRC_DIR):
            os.makedirs(SRC_DIR)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        for name in ['train.dst', 'valid.dst', 'train.src', 'valid.src']:
        # for name in ['test.dst', 'test.src']:
            with open(SRC_DIR+"/"+name) as f, open(DATA_DIR+"/"+name, 'w') as g:
                for x in f:
                    x = x.strip()
                    x = re.sub(r'\s+', ' ', x)
                    x = sp.encode_as_pieces(x)
                    x = ' '.join(x)
                    print(x, file=g)
        print(DATANAME, end=" ", flush=True)
    except:
        pass